import numpy as np
from collections import namedtuple
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from sentence_transformers import SentenceTransformer

# adapted from https://github.com/pytorch/examples/blob/master/reinforcement_learning/actor_critic.py


def actor_critic(mwg, model_parameters, training_parameters, base_path, logger, save_results):
    SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

    running_reward = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    available_actions = mwg.total_available_actions

    emsize = model_parameters['embedding_size']  # embedding size of the bert model
    max_sequence_length = model_parameters['max_sequence_length']    # maximum length the text state of the env will get padded to
    output_size = len(available_actions)
    num_layers = model_parameters['num_layers']
    action_model = ActionModel(emsize,
                               max_sequence_length,
                               output_size,
                               num_layers).to(device)

    value_model = ValueModel(emsize,
                             max_sequence_length,
                             1,
                             num_layers).to(device)

    lr = training_parameters['learning_rate']
    num_episodes = training_parameters['num_episodes']
    batch_size = training_parameters['batch_size']
    gamma = training_parameters['gamma']
    max_steps = training_parameters['max_steps']
    checkpoint_frequency = training_parameters['checkpoint_frequency']  # how often should a checkpoint be created
    starting_episode = 0

    action_optimizer = torch.optim.Adam(action_model.parameters(), lr=lr)
    value_optimizer = torch.optim.Adam(action_model.parameters(), lr=lr)

    em_model = SentenceTransformer(model_parameters['embedding_model'])

    eps = np.finfo(np.float32).eps.item()

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
            im_tensor = torch.FloatTensor([im]).to(device)

            text = state['text_state']
            embeddings = em_model.encode(text)
            embedded_text_tensor = torch.FloatTensor([embeddings]).to(device)

            action_probabilities = action_model(im_tensor, embedded_text_tensor)
            logger.debug('action_probabilities',action_probabilities)
            state_value = value_model(im_tensor, embedded_text_tensor)
            # action_probabilities = action_probabilities.cpu().detach().numpy()[0]

            # create a categorical distribution over the list of probabilities of actions
            m = Categorical(action_probabilities)
            # and sample an action using the distribution
            action = m.sample()
            logger.debug('SavedAction', type(SavedAction), SavedAction)

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
                    print('---------training---------')

                    # TODO ignore for now
                    # update cumulative reward
                    # running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

                    # perform backprop
                    R = 0
                    policy_losses = []  # list to save actor (policy) loss
                    value_losses = []  # list to save critic (value) loss
                    returns = []  # list to save the true values

                    # calculate the true value using rewards returned from the environment
                    for r in batch_rewards[::-1]:
                        # calculate the discounted value
                        R = r + gamma * R
                        returns.insert(0, R)

                    returns = torch.tensor(returns).to(device)
                    returns = (returns - returns.mean()) / (returns.std() + eps)

                    for (log_prob, value), R in zip(batch_actions, returns):
                        advantage = R - value.item()

                        # calculate actor (policy) loss
                        policy_losses.append(-log_prob * advantage)

                        # calculate critic (value) loss using L1 smooth loss
                        value_losses.append(F.smooth_l1_loss(value, torch.tensor([R]).to(device)))

                    # reset gradients
                    action_optimizer.zero_grad()
                    value_optimizer.zero_grad()
                    # sum up all the values of policy_losses and value_losses
                    action_loss = torch.stack(policy_losses).sum()
                    value_loss = torch.stack(value_losses).sum()

                    # perform backprop
                    action_loss.backward()
                    value_loss.backward()
                    action_optimizer.step()
                    value_optimizer.step()

                    batch_rewards = []
                    batch_actions = []
                    batch_counter = 0

                    # log results
                    # if episode % args.log_interval == 0:
                    #     print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                    #           episode, ep_reward, running_reward))

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


class ValueModel(nn.Module):
    def __init__(self, emsize, max_sequence_length, output_size, num_layers):
        super(ValueModel, self).__init__()
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
        return output

    def init_weights(self):
        # TODO add init for cnn
        # make initrange a parameter
        initrange = 0.1
        # self.encoder.weight.data.uniform_(-initrange, initrange)
        # self.decoder.bias.data.zero_()
        # self.decoder.weight.data.uniform_(-initrange, initrange)