import numpy as np
from tqdm import tqdm
import os

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import DataParallel
from torchvision import transforms

from utils.distances import load_inception
from sentence_transformers import SentenceTransformer


# adapted from: https://towardsdatascience.com/learning-reinforcement-learning-reinforce-with-pytorch-5e8ad7fc7da0
def reinforce(mwg, model_parameters, training_parameters, base_path,
              logger, save_model, gpu, load_model):
    """

    Args:
        gpu:
        load_model:
        save_model:
        model_parameters:
        training_parameters:
        base_path:
        mwg:
        logger:

    Returns:

    """
    # TODO include flag to turn on/off training, so model doesn't train on eval data

    available_actions = mwg.total_available_actions
    action_space = np.arange(len(available_actions))

    # assign all available gpu devices to pytorch
    device = torch.device(gpu if torch.cuda.is_available() else "cpu")
    logger.debug(f'Device: {device}')

    model = DataParallel(RLBaseline(model_parameters['embedding_size'], model_parameters['hidden_layer_size'],
                                    output_size=len(available_actions)).to(device))

    inception = load_inception(device)
    preprocess = transforms.Compose([
        transforms.Resize(299),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    lr = training_parameters['learning_rate']
    num_episodes = int(training_parameters['num_episodes'])
    batch_size = training_parameters['batch_size']
    gamma = training_parameters['gamma']
    max_steps = training_parameters['max_steps']
    checkpoint_frequency = training_parameters['checkpoint_frequency']  # how often should a checkpoint be created
    starting_episode = 0

    # TODO maybe also use lr scheduler (adjust lr if) // Gradient clipping
    # TODO record loss function in parameters ?
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # TODO increase em model sentence max length
    # TODO look into what the model was trained on. How does it deal with multiple sentences ?
    em_model = SentenceTransformer(model_parameters['embedding_model'])

    ck_path = os.path.join(base_path, 'checkpoint.pt')

    if os.path.isfile(ck_path) and load_model:
        # if a checkpoint for the model already exist resume from there
        checkpoint = torch.load(ck_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        starting_episode = checkpoint['current_episode']

    if not os.path.isdir(base_path) and save_model:
        os.makedirs(base_path)

    total_rewards = []
    total_steps = []
    hits = []

    batch_rewards = []
    batch_actions = []
    batch_states_image = []
    batch_states_text = []
    batch_counter = 0

    for episode in tqdm(range(starting_episode, num_episodes)):

        state = mwg.reset()

        states_image = []
        states_text = []
        rewards = []
        actions = []

        done = False
        steps = 0

        while not done and steps < max_steps:

            # preprocess state (image and text)
            processed_frame = preprocess(state['current_room']).unsqueeze(0).to(device)
            with torch.no_grad():
                im = inception(processed_frame).squeeze().cpu().detach().numpy()
            im_tensor = torch.FloatTensor([im])

            embeddings = em_model.encode(state['text_state'])
            embedded_text_tensor = torch.FloatTensor([embeddings])

            logger.debug(embedded_text_tensor)
            action_probabilities = model(im_tensor.to(device),
                                         embedded_text_tensor.to(device))
            action_probabilities = action_probabilities.cpu().detach().numpy()[0]
            action = np.random.choice(action_space, p=action_probabilities)

            state, reward, done, room_found = mwg.step(action)

            states_image.append(im)
            states_text.append(embeddings)
            rewards.append(reward)
            actions.append(action)

            steps += 1
            if done or steps >= max_steps:
                # save the results for the episode
                total_rewards.append(mwg.model_return)
                total_steps.append(steps)
                hits.append(room_found)

                # when a episode is finished, collect experience
                batch_rewards.extend(discount_rewards(rewards, gamma))
                batch_states_image.extend(states_image)
                batch_states_text.extend(states_text)
                batch_actions.extend(actions)
                batch_counter += 1

                if batch_counter == batch_size:

                    model.train()
                    optimizer.zero_grad()

                    # cast the batch to tensors and onto the GPU
                    im_tensor = torch.FloatTensor(batch_states_image).to(device)
                    inputs_tensor = torch.FloatTensor(batch_states_text).to(device)
                    reward_tensor = torch.FloatTensor(batch_rewards).to(device)
                    action_tensor = torch.LongTensor(batch_actions).to(device)

                    logprob = torch.log(model(im_tensor, inputs_tensor))
                    selected_logprobs = reward_tensor * torch.gather(logprob, 1, action_tensor.unsqueeze(1)).squeeze()
                    loss = -selected_logprobs.mean()

                    # Calculate gradients
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    # Apply gradients
                    optimizer.step()

                    batch_rewards = []
                    batch_actions = []
                    batch_states_image = []
                    batch_states_text = []
                    batch_counter = 0

        # save the progress of the training every checkpoint_frequency episodes
        if episode % checkpoint_frequency == 0 and save_model:
            torch.save({
                'current_episode': episode,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, ck_path)

    return total_rewards, total_steps, hits


def discount_rewards(rewards, gamma=0.99):
    """

    Args:
        rewards:
        gamma:

    Returns:

    """
    r = np.array([gamma**i * rewards[i] for i in range(len(rewards))])
    # Reverse the array direction for cumsum and then
    # revert back to the original order
    r = r[::-1].cumsum()[::-1]
    return r - r.mean()


class RLBaseline(nn.Module):
    def __init__(self, emsize, hidden_layer_size, output_size):
        super(RLBaseline, self).__init__()

        # TODO with the sentence transformer max_sequence_length is not a thing anymore
        # TODO maybe replace it with something else (emsize, etc)
        # TODO https://arxiv.org/pdf/1902.07742.pdf

        self.fc_image = nn.Linear(2048, hidden_layer_size)

        self.fc_text = nn.Linear(emsize, hidden_layer_size)

        self.fc0 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.fc1 = nn.Linear(hidden_layer_size, output_size)

    def forward(self, image, text):
        image = torch.tanh(self.fc_image(image))

        text = torch.tanh(self.fc_text(text))

        # combine image and text into one vector
        output = torch.mul(image, text)

        output = F.relu(self.fc4(output))
        output = self.fc5(output)

        actions = F.softmax(output, dim=1)
        return actions
