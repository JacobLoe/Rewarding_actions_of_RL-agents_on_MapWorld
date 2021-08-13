import os
import numpy as np
from tqdm import tqdm
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from sentence_transformers import SentenceTransformer


# adapted from https://github.com/pytorch/examples/blob/master/reinforcement_learning/actor_critic.py
def actor_critic(mwg, model_parameters, training_parameters, base_path, logger, save_model, gpu, load_model):
    running_reward = 10
    SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

    device = torch.device(gpu if torch.cuda.is_available() else "cpu")
    available_actions = mwg.total_available_actions

    emsize = model_parameters['embedding_size']  # embedding size of the bert model
    max_sequence_length = model_parameters['max_sequence_length']    # maximum length the text state of the env will get padded to
    output_size = len(available_actions)
    num_layers = model_parameters['num_layers']
    model = ActionModel(emsize,
                        max_sequence_length,
                        output_size,
                        num_layers).to(device)

    lr = training_parameters['learning_rate']
    num_episodes = int(training_parameters['num_episodes'])
    batch_size = training_parameters['batch_size']
    gamma = training_parameters['gamma']
    max_steps = training_parameters['max_steps']
    checkpoint_frequency = training_parameters['checkpoint_frequency']  # how often should a checkpoint be created
    starting_episode = 0

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    em_model = SentenceTransformer(model_parameters['embedding_model'])

    eps = np.finfo(np.float32).eps.item()

    ck_path = os.path.join(base_path, 'checkpoint.pt')

    if os.path.isfile(ck_path) and load_model:
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
    batch_counter = 0

    for episode in tqdm(range(starting_episode, num_episodes)):

        # reset environment and episode reward
        state = mwg.reset()

        saved_actions = []
        rewards = []

        done = False
        steps = 0

        while not done and steps < max_steps:

            im = state['current_room']
            im = np.reshape(im, (np.shape(im)[2], np.shape(im)[1], np.shape(im)[0]))
            im_tensor = torch.FloatTensor([im])

            text = state['text_state']
            embeddings = em_model.encode(text)
            embedded_text_tensor = torch.FloatTensor([embeddings])

            action_probabilities, state_value = model(im_tensor.to(device),
                                                      embedded_text_tensor.to(device))
            logger.debug('action_probabilities', action_probabilities)

            # create a categorical distribution over the list of probabilities of actions
            m = Categorical(action_probabilities)
            # and sample an action using the distribution
            action = m.sample()

            # save to action buffer
            saved_actions.append(SavedAction(m.log_prob(action), state_value))

            # take the action
            state, reward, done, room_found = mwg.step(action.item())

            rewards.append(reward)

            steps += 1

            if done or steps >= max_steps:
                # save the results for the episode
                total_rewards.append(mwg.model_return)
                total_steps.append(steps)
                hits.append(room_found)

                # when a episode is finished, collect experience
                batch_rewards.extend(rewards)
                batch_actions.extend(saved_actions)
                batch_counter += 1
                if batch_counter == batch_size:

                    # TODO ignore for now
                    # update cumulative reward
                    # running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

                    # perform backprop
                    R = 0
                    policy_losses = []  # list to save actor (policy) loss
                    value_losses = []  # list to save critic (value) loss
                    returns = []  # list to save the true values

                    # TODO check whether this and discount_rewards amount to the same operation
                    # calculate the true value using rewards returned from the environment
                    for r in batch_rewards[::-1]:
                        # calculate the discounted value
                        R = r + gamma * R
                        returns.insert(0, R)

                    returns = torch.tensor(returns, dtype=torch.float32).to(device)
                    returns = (returns - returns.mean()) / (returns.std() + eps)

                    for (log_prob, value), R in zip(batch_actions, returns):
                        advantage = R - value.item()

                        # calculate actor (policy) loss
                        policy_losses.append(-log_prob * advantage)

                        # calculate critic (value) loss using L1 smooth loss
                        value_losses.append(F.smooth_l1_loss(value, torch.tensor([R]).unsqueeze(-1).to(device)))

                    # reset gradients
                    optimizer.zero_grad()
                    # sum up all the values of policy_losses and value_losses
                    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

                    # perform backprop
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    optimizer.step()

                    batch_rewards = []
                    batch_actions = []
                    batch_counter = 0

                    # log results
                    # if episode % args.log_interval == 0:
                    #     print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                    #           episode, ep_reward, running_reward))

        # save the progress of the training every checkpoint_frequency episodes
        if episode % checkpoint_frequency == 0 and save_model:
            torch.save({
                'current_episode': episode,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, ck_path)

    return total_rewards, total_steps, hits


class ActionModel(nn.Module):
    def __init__(self, emsize, max_sequence_length, output_size, num_layers):
        super(ActionModel, self).__init__()
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

        # action model
        self.fc4_action = nn.Linear(max_sequence_length, max_sequence_length)
        self.fc5_action = nn.Linear(max_sequence_length, output_size)

        # value model
        self.fc4_value = nn.Linear(max_sequence_length, max_sequence_length)
        self.fc5_value = nn.Linear(max_sequence_length, 1)

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

        # combine image and text into one vector
        output = torch.mul(cnn, text)

        # compute the best action for a state
        actions = F.relu(self.fc4_action(output))
        actions = self.fc5_action(actions)
        actions = F.softmax(actions, dim=1)

        # compute the value of being in a state
        value = F.relu(self.fc4_value(output))
        value = self.fc5_value(value)

        return actions, value

    def init_weights(self):
        # TODO add init for cnn
        # make initrange a parameter
        initrange = 0.1
        # self.encoder.weight.data.uniform_(-initrange, initrange)
        # self.decoder.bias.data.zero_()
        # self.decoder.weight.data.uniform_(-initrange, initrange)