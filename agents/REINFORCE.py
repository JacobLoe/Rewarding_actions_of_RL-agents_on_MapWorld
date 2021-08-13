import numpy as np
from tqdm import tqdm
import os
from collections import namedtuple

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import DataParallel
from torch.utils.data import DataLoader

from utils import preprocess_mapworld_state
from utils.ReplayBuffer import ReplayBuffer, RLDataset
from sentence_transformers import SentenceTransformer


# adapted from: https://towardsdatascience.com/learning-reinforcement-learning-reinforce-with-pytorch-5e8ad7fc7da0
def reinforce(mwg, model_parameters, training_parameters, base_path, logger, save_model, gpu):
    """

    Args:
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
    # TODO switch to gym action space
    action_space = np.arange(len(available_actions))

    # assign all available gpu devices to pytorch
    device_count = [i for i in range(torch.cuda.device_count())]
    logger.debug('device count', device_count)
    device = torch.device(gpu if torch.cuda.is_available() else "cpu")
    logger.debug(f'Device: {device}')

    output_size = len(available_actions)
    model = DataParallel(RLBaseline(model_parameters['embedding_size'],
                                    model_parameters['max_sequence_length'],
                                    output_size,
                                    model_parameters['num_layers']).to(device))

    buffer = ReplayBuffer(training_parameters['replay_size'])
    # Named tuple for storing experience steps gathered during training
    Experience = namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])

    lr = training_parameters['learning_rate']
    num_episodes = training_parameters['num_episodes']
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

    if os.path.isdir(ck_path) and save_model:
        # if a checkpoint for the model already exist resume from there
        checkpoint = torch.load(ck_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        starting_episode = checkpoint['current_episode']
    elif not os.path.isdir(base_path) and save_model:
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

        s_0 = mwg.reset()

        states_image = []
        states_text = []
        rewards = []
        actions = []

        done = False
        steps = 0

        while not done and steps < max_steps:
            # preprocess state (image and text)
            im, embeddings = preprocess_mapworld_state(s_0, em_model=em_model)
            im_tensor = torch.FloatTensor([im])
            embedded_text_tensor = torch.FloatTensor([embeddings])

            action_probabilities = model(im_tensor.to(device),
                                         embedded_text_tensor.to(device))
            action_probabilities = action_probabilities.cpu().detach().numpy()[0]
            action = np.random.choice(action_space, p=action_probabilities)

            s_1, reward, done, room_found = mwg.step(action)

            exp = Experience(s_0, action, reward, done, [])
            buffer.append(exp)

            states_image.append(im)
            states_text.append(embeddings)
            rewards.append(reward)
            actions.append(action)

            s_0 = s_1
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
                    dataset = RLDataset(buffer, training_parameters['max_steps'])
                    dataloader = DataLoader(dataset=dataset,
                                            batch_size=training_parameters['batch_size'])
                    for i, d in enumerate(dataloader):
                        print(i, d[0])
                    print(sdsadsd)

                    model.train()
                    optimizer.zero_grad()

                    # cast the batch to tensors and onto the GPU
                    im_tensor = torch.FloatTensor(batch_states_image).to(device)
                    inputs_tensor = torch.FloatTensor(batch_states_text).to(device)
                    reward_tensor = torch.FloatTensor(batch_rewards).to(device)
                    action_tensor = torch.LongTensor(batch_actions).to(device)
                    print('im_tensor', im_tensor.size())

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
    def __init__(self, emsize, max_sequence_length, output_size, num_layers):
        super(RLBaseline, self).__init__()

        # TODO replace cnn and lstm with pretrained pytorch models

        # TODO with the sentence transformer max_sequence_length is not a thing anymore
        # TODO maybe replace it with something else (emsize, etc)
        # TODO https://arxiv.org/pdf/1902.07742.pdf

        # TODO look into padding of input image
        # CNN
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc_cnn1 = nn.Linear(121104, 1200)   # layer size is result of image res (360x360) after conv + pool

        self.fc_cnn2 = nn.Linear(1200, max_sequence_length)

        # text processing
        self.lstm1 = nn.LSTM(1, emsize, batch_first=True, num_layers=num_layers)
        self.fc_lstm = nn.Linear(emsize, max_sequence_length)

        self.fc4 = nn.Linear(max_sequence_length, max_sequence_length)
        self.fc5 = nn.Linear(max_sequence_length, output_size)

        # self.init_weights()

    def forward(self, im, text):
        cnn = self.pool(F.relu(self.conv1(im)))
        cnn = self.pool(F.relu(self.conv2(cnn)))
        cnn = torch.flatten(cnn, 1)     # flatten all dimensions except batch
        cnn = F.relu(self.fc_cnn1(cnn))
        cnn = torch.tanh(self.fc_cnn2(cnn))

        text, _ = self.lstm1(text.unsqueeze(-1))

        text = torch.tanh(self.fc_lstm(text))
        text = torch.mean(text, dim=1)

        output = torch.mul(cnn, text)
        output = F.relu(self.fc4(output))
        output = self.fc5(output)
        actions = F.softmax(output, dim=1)
        return actions

    def init_weights(self):
        # TODO add init for cnn
        # make initrange a parameter
        initrange = 0.1
        # self.encoder.weight.data.uniform_(-initrange, initrange)
        # self.decoder.bias.data.zero_()
        # self.decoder.weight.data.uniform_(-initrange, initrange)