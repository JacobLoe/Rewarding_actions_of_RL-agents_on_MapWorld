import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import os
from time import time


# adapted from: https://towardsdatascience.com/learning-reinforcement-learning-reinforce-with-pytorch-5e8ad7fc7da0
def reinforce(mwg, model_parameters, training_parameters, base_path, logger):
    """

    Args:
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.debug('Devive: {}'.format(device))
    emsize = model_parameters['embedding_size']  # embedding size of the bert model
    max_sequence_length = model_parameters['max_sequence_length']    # maximum length the text state of the env will get padded to
    model = RLBaseline(emsize, max_sequence_length, len(available_actions)).to(device)

    lr = training_parameters['learning_rate']
    num_episodes = training_parameters['num_episodes']
    batch_size = training_parameters['batch_size']
    gamma = training_parameters['gamma']
    max_steps = training_parameters['max_steps']
    checkpoint_frequency = training_parameters['checkpoint_frequency']  # how often should a checkpoint be created
    starting_episode = 0

    # TODO look into different loss functions ADAM ?
    # TODO maybe also use lr scheduler (adjust lr if) // Gradient clipping
    # TODO record loss function in parameters ?
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # TODO look into what the model was trained on. How does it deal with multiple sentences ?
    em_model = SentenceTransformer(model_parameters['embedding_model'])

    ck_path = os.path.join(base_path, 'checkpoint.pt')

    if os.path.isdir(ck_path):
        # if a checkpoint for the model already exist resume from there
        checkpoint = torch.load(ck_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        starting_episode = checkpoint['current_episode']
    if not os.path.isdir(base_path):
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

        t_pr = time()
        s_0 = mwg.reset()
        logger.debug(f'Time for env reset: {time()-t_pr}')

        states_image = []
        states_text = []
        rewards = []
        actions = []

        done = False
        steps = 0
        # loop until an episode is finished or 30 steps have been done
        # 30 steps is worse than a agent with random actions
        while not done and steps < max_steps:

            t_pp = time()
            # preprocess state (image and text)
            im = s_0['current_room']
            im = np.reshape(im, (np.shape(im)[2], np.shape(im)[1], np.shape(im)[0]))
            im_tensor = torch.FloatTensor([im]).to(device)
            logger.debug(f'Time for image preprocessing: {time()-t_pp}')

            t_ppt = time()
            text = s_0['question'] + ' ' + s_0['directions']
            embeddings = em_model.encode(text)
            embedded_text_tensor = torch.FloatTensor([embeddings]).to(device)
            logger.debug(f'Time for text embedding: {time()-t_ppt}')

            t_ga = time()
            action_probabilities = model(im_tensor,
                                         embedded_text_tensor)
            logger.debug(f'Time to get action probs from model: {time()-t_ga}')
            t_ca = time()
            action_probabilities = action_probabilities.cpu().detach().numpy()[0]
            action = np.random.choice(action_space, p=action_probabilities)
            logger.debug(f'Time to choose action: {time()-t_ca}')

            t_s = time()
            s_1, reward, done, room_found = mwg.step(action)
            logger.debug(f'Time for env step: {time()-t_s}')

            states_image.append(im)
            states_text.append(embeddings)
            rewards.append(reward)
            actions.append(action)

            s_0 = s_1
            steps += 1
            if done or steps >= max_steps:
                t_srb = time()
                # save the results for the episode
                total_rewards.append(mwg.model_return)
                total_steps.append(steps)
                hits.append(room_found)

                # when a episode is finished, collect experience
                batch_rewards.extend(discount_rewards(
                    rewards, gamma))
                batch_states_image.extend(states_image)
                batch_states_text.extend(states_text)
                batch_actions.extend(actions)
                batch_counter += 1
                logger.debug(f'Time for saving batch: {time()-t_srb}')

                if batch_counter == batch_size:
                    print('---------training---------')
                    model.train()
                    optimizer.zero_grad()

                    t_ctt = time()
                    # cast the batch to tensors and onto the GPU
                    im_tensor = torch.FloatTensor(batch_states_image).to(device)
                    inputs_tensor = torch.FloatTensor(batch_states_text).to(device)
                    reward_tensor = torch.FloatTensor(batch_rewards).to(device)
                    action_tensor = torch.LongTensor(batch_actions).to(device)
                    logger.debug(f'Time to cast batches to tensors: {time()-t_ctt}')

                    t_lp = time()
                    logprob = torch.log(model(im_tensor, inputs_tensor))
                    selected_logprobs = reward_tensor * \
                                        torch.gather(logprob, 1, action_tensor.unsqueeze(1)).squeeze()
                    loss = -selected_logprobs.mean()
                    logger.debug(f'Time for loss calc: {time()-t_lp}')

                    t_bp = time()
                    # Calculate gradients
                    loss.backward()
                    logger.debug(f'Time for backprop {time()-t_bp}')

                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    # Apply gradients
                    optimizer.step()

                    batch_rewards = []
                    batch_actions = []
                    batch_states_image = []
                    batch_states_text = []
                    batch_counter = 1

        logger.debug(f'Time for an full episode: {time()-t_pr} \n')

        # save the progress of the training every checkpoint_frequency episodes
        if episode % checkpoint_frequency == 0:
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
    r = np.array([gamma**i * rewards[i]
        for i in range(len(rewards))])
    # Reverse the array direction for cumsum and then
    # revert back to the original order
    r = r[::-1].cumsum()[::-1]
    return r - r.mean()


class RLBaseline(nn.Module):
    def __init__(self, emsize, max_sequence_length, output_size):
        super(RLBaseline, self).__init__()

        # TODO look into padding of input image
        # CNN
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(121104, 1200)   # layer size is result of image res (360x360) after conv + pool
        # TODO with the sentence transformer max_sequence_length is not a thing anymore
        # TODO maybe replace it with something else (emsize, etc)
        # TODO use tanh for both last fc like in https://arxiv.org/pdf/1902.07742.pdf
        self.fc2 = nn.Linear(1200, max_sequence_length)

        # text processing
        self.fc3 = nn.Linear(emsize, max_sequence_length)

        self.fc4 = nn.Linear(max_sequence_length, output_size)

        # self.init_weights()

    def forward(self, im, text):
        cnn = self.pool(F.relu(self.conv1(im)))
        cnn = self.pool(F.relu(self.conv2(cnn)))
        cnn = torch.flatten(cnn, 1)     # flatten all dimensions except batch
        cnn = F.relu(self.fc1(cnn))
        cnn = F.relu(self.fc2(cnn))

        text = self.fc3(text)

        output = torch.mul(cnn, text)
        output = self.fc4(output)
        actions = F.softmax(output, dim=1)
        return actions

    def init_weights(self):
        # TODO add init for cnn
        # make initrange a parameter
        initrange = 0.1
        # self.encoder.weight.data.uniform_(-initrange, initrange)
        # self.decoder.bias.data.zero_()
        # self.decoder.weight.data.uniform_(-initrange, initrange)