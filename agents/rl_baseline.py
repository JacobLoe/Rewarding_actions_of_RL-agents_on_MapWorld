import numpy as np
import gym
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math


# https://towardsdatascience.com/learning-reinforcement-learning-reinforce-with-pytorch-5e8ad7fc7da0
class PolicyEstimator:
    def __init__(self, env):
        self.n_inputs = env.observation_space.shape[0]
        self.n_outputs = env.action_space.n

        # Define network
        self.network = nn.Sequential(
            nn.Linear(self.n_inputs, 16),
            nn.ReLU(),
            nn.Linear(16, self.n_outputs),
            nn.Softmax(dim=-1))

    def predict(self, state):
        action_probs = self.network(torch.FloatTensor(state))
        return action_probs


def discount_rewards(rewards, gamma=0.99):
    r = np.array([gamma**i * rewards[i]
        for i in range(len(rewards))])
    # Reverse the array direction for cumsum and then
    # revert back to the original order
    r = r[::-1].cumsum()[::-1]
    return r - r.mean()


def reinforce(env, policy_estimator, num_episodes=2000,
              batch_size=10, gamma=0.99):
    # Set up lists to hold results
    total_rewards = []
    batch_rewards = []
    batch_actions = []
    batch_states = []
    batch_counter = 1

    # Define optimizer
    optimizer = optim.Adam(policy_estimator.network.parameters(),
                           lr=0.01)

    action_space = np.arange(env.action_space.n)
    print('action space', action_space)
    ep = 0
    while ep < num_episodes:
        s_0 = env.reset()
        states = []
        rewards = []
        actions = []
        done = False
        while done == False:
            # Get actions and convert to numpy array
            action_probs = policy_estimator.predict(
                s_0).detach().numpy()
            action = np.random.choice(action_space,
                                      p=action_probs)
            s_1, r, done, _ = env.step(action)

            states.append(s_0)
            rewards.append(r)
            actions.append(action)
            s_0 = s_1

            # If done, batch data
            if done:
                batch_rewards.extend(discount_rewards(
                    rewards, gamma))
                batch_states.extend(states)
                batch_actions.extend(actions)
                batch_counter += 1
                total_rewards.append(sum(rewards))

                # If batch is complete, update network
                if batch_counter == batch_size:
                    optimizer.zero_grad()
                    state_tensor = torch.FloatTensor(batch_states)
                    reward_tensor = torch.FloatTensor(
                        batch_rewards)
                    # Actions are used as indices, must be
                    # LongTensor
                    action_tensor = torch.LongTensor(
                        batch_actions)
                    # print('actions', type(batch_actions), np.shape(batch_actions), type(action_tensor), np.shape(action_tensor))
                    # print(batch_actions)
                    # print(action_tensor)
                    # Calculate loss
                    print('model ', policy_estimator.predict(state_tensor))
                    logprob = torch.log(
                        policy_estimator.predict(state_tensor))
                    print('logprob', logprob, np.shape(logprob))
                    # selected_logprobs = reward_tensor * \
                    #                     torch.gather(logprob, 1,
                    #                                  action_tensor.unsqueeze(1)).squeeze()
                    a = action_tensor.unsqueeze(1)
                    print(a, np.shape(a))
                    b = torch.gather(logprob, 1, a)
                    print(b)
                    c = b.squeeze()
                    print(c)
                    selected_logprobs = reward_tensor * \
                                        c
                    loss = -selected_logprobs.mean()

                    # Calculate gradients
                    loss.backward()
                    # Apply gradients
                    optimizer.step()

                    batch_rewards = []
                    batch_actions = []
                    batch_states = []
                    batch_counter = 1

                avg_rewards = np.mean(total_rewards[-100:])
                # Print running average
                print("\rEp: {} Average of last 100: {:.2f}".format(ep + 1, avg_rewards), end="")
                ep += 1

    return total_rewards


class Net(nn.Module):
    def __init__(self, ntoken, emsize, nhead, nhid, nlayers, dropout, max_sequence_length, output_size):
        super(Net, self).__init__()

        # CNN
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(393120, 1200)   # layer size is result of image res (480x854) after conv + pool
        self.fc2 = nn.Linear(1200, 840)
        self.fc3 = nn.Linear(840, max_sequence_length)

        # Transformer
        self.encoder = nn.Embedding(ntoken, emsize)
        self.pos_encoder = PositionalEncoding(emsize, dropout)
        encoder_layers = TransformerEncoderLayer(emsize, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        # self.decoder = nn.Linear(200, ntoken)

        self.fc4 = nn.Linear(2*max_sequence_length, output_size)

        self.ntoken = ntoken
        self.ninp = emsize
        self.init_weights()

    def forward(self, im, src, src_mask):
        x = self.pool(F.relu(self.conv1(im)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)     # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = output.mean(dim=2)
        # output = self.decoder(output)

        # TODO rename output
        z = torch.cat((x, output), dim=1)

        z = self.fc4(z)
        z = F.softmax(z, dim=1)
        return z

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        # TODO add init for cnn
        # make initrange a parameter
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        # self.decoder.bias.data.zero_()
        # self.decoder.weight.data.uniform_(-initrange, initrange)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


if __name__ == '__main__':

    env = gym.make('CartPole-v0')
    policy_est = PolicyEstimator(env)
    rewards = reinforce(env, policy_est)
