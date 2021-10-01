from utils.ReplayBuffer import ReplayBuffer, RLDataset
from torch import nn
import gym
import torch
import pytorch_lightning as pl
from collections import OrderedDict, namedtuple
from typing import Tuple, List
import torch.optim as optim
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import numpy as np
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer


# Named tuple for storing experience steps gathered in training
Experience = namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])


class DQN(pl.LightningModule):#nn.Module):
    def __init__(self, emsize, max_sequence_length, output_size, num_layers):
        super(DQN, self).__init__()

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
        o = self.conv1(im)
        p = F.relu(o)
        cnn = self.pool(p)
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
        # TODO remove softmax, just use fc-layer
        actions = F.softmax(output, dim=1)
        return actions


def preprocess_mapworld_state(state, em_model):
    im = state['current_room']
    im = np.reshape(im, (np.shape(im)[2], np.shape(im)[1], np.shape(im)[0]))

    text = state['text_state']
    embeddings = em_model.encode(text)
    return im, embeddings


class Agent:
    """
    Base Agent class handling the interaction with the environment

    Args:
        env: training environment
        replay_buffer: replay buffer storing experiences
    """

    def __init__(self, env: gym.Env, replay_buffer: ReplayBuffer, embedding_model) -> None:
        self.env = env
        self.replay_buffer = replay_buffer
        self.reset()
        self.state = self.env.reset()
        self.em_model = SentenceTransformer(embedding_model)

    def reset(self) -> None:
        """ Resents the environment and updates the state"""
        self.state = self.env.reset()

    def get_action(self, net: nn.Module, epsilon: float, device: str) -> int:
        """
        Using the given network, decide what action to carry out
        using an epsilon-greedy policy

        Args:
            net: DQN network
            epsilon: value to determine likelihood of taking a random action
            device: current device

        Returns:
            action
        """

        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:

            im, embeddings = preprocess_mapworld_state(state=self.state, em_model=self.em_model)
            im_tensor = torch.FloatTensor([im])
            embedded_text_tensor = torch.FloatTensor([embeddings])
            if device not in ['cpu']:
                print('device', device)
                im_tensor = im_tensor#.cuda(device)
                embedded_text_tensor = embedded_text_tensor#.cuda(device)

            q_values = net(im_tensor, embedded_text_tensor)
            _, action = torch.max(q_values, dim=1)
            action = int(action.item())

        return action

    @torch.no_grad()
    def play_step(self, net: nn.Module, epsilon: float = 0.0, device: str = 'cpu') -> Tuple[float, bool, int]:
        """
        Carries out a single interaction step between the agent and the environment

        Args:
            net: DQN network
            epsilon: value to determine likelihood of taking a random action
            device: current device

        Returns:
            reward, done
        """

        action = self.get_action(net, epsilon, device)

        # do step in the environment
        new_state, reward, done, room_found = self.env.step(action)

        # print('\n self.state', type(self.state), np.shape(self.state['current_room']))
        # print('new_state', type(new_state), np.shape(new_state['current_room']))

        im, embeddings = preprocess_mapworld_state(new_state, self.em_model)
        temp_new_state = {'current_room': im, 'text_state': embeddings}

        im, embeddings = preprocess_mapworld_state(self.state, self.em_model)
        old_state = {'current_room': im, 'text_state': embeddings}

        exp = Experience(old_state, action, reward, done, temp_new_state)

        self.replay_buffer.append(exp)

        self.state = new_state
        if done:
            self.reset()
        return reward, done, room_found


class DQNLightning(pl.LightningModule):
    """ Basic DQN Model """

    def __init__(self, mwg, model_parameters, training_parameters) -> None:
        super().__init__()
        self.training_parameters = training_parameters
        self.model_parameters = model_parameters

        self.env = mwg
        n_actions = self.env.action_space.n

        self.net = DQN(model_parameters['embedding_size'],
                       model_parameters['max_sequence_length'],
                       n_actions,
                       model_parameters['num_layers'])
        self.target_net = DQN(model_parameters['embedding_size'],
                              model_parameters['max_sequence_length'],
                              n_actions,
                              model_parameters['num_layers'])

        self.buffer = ReplayBuffer(self.training_parameters['replay_size'])
        self.agent = Agent(self.env, self.buffer, model_parameters['embedding_model'])
        self.total_reward = 0
        self.episode_reward = 0
        self.populate(self.training_parameters['warm_start_steps'])

    def populate(self, steps: int = 1000) -> None:
        """
        Carries out several random steps through the environment to initially fill
        up the replay buffer with experiences

        Args:
            steps: number of random steps to populate the buffer with
        """
        for i in range(steps):
            self.agent.play_step(self.net, epsilon=1.0)

    def forward(self, im: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        """
        Passes in a state x through the network and gets the q_values of each action as an output

        Args:
            im: environment state
            text: environment state
        Returns:
            q values
        """
        output = self.net(im, text)
        return output

    # todo
    def dqn_mse_loss(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Calculates the mse loss using a mini batch from the replay buffer

        Args:
            batch: current mini batch of replay data

        Returns:
            loss
        """
        states, actions, rewards, dones, next_states = batch
        # print('\ndqn_mse_loss states shape', np.shape(states['current_room']))

        print('\nimage', states['current_room'].device)
        print('text', states['text_state'].device)

        device = self.get_device([states])
        # # self.net = self.net.to(device)
        # print(self.net.device)
        # print(device)
        # self.net.cuda(device)
        print(self.net.device)

        state_action_values = self.net(states['current_room'], states['text_state'])

        state_action_values = state_action_values.gather(1, actions.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            next_state_values = self.target_net(next_states['current_room'],
                                                next_states['text_state']).max(1)[0]
            next_state_values[dones] = 0.0
            next_state_values = next_state_values.detach()

        expected_state_action_values = next_state_values * self.hparams.gamma + rewards

        return nn.MSELoss()(state_action_values, expected_state_action_values)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], nb_batch) -> OrderedDict:
        """
        Carries out a single step through the environment to update the replay buffer.
        Then calculates loss based on the minibatch recieved

        Args:
            batch: current mini batch of replay data
            nb_batch: batch number

        Returns:
            Training loss and log metrics
        """
        device = self.get_device(batch)
        self.net.cuda(device)
        self.target_net.cuda(device)
        epsilon = max(self.training_parameters['eps_end'], self.training_parameters['eps_start'] -
                      self.global_step + 1 / self.training_parameters['eps_last_frame'])

        # step through environment with agent
        reward, done, room_found = self.agent.play_step(self.net, epsilon, device)
        self.episode_reward += reward

        # calculates training loss
        loss = self.dqn_mse_loss(batch)

        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss = loss.unsqueeze(0)

        if done:
            self.total_reward = self.episode_reward
            self.episode_reward = 0

        # Soft update of target network
        if self.global_step % self.training_parameters['sync_rate'] == 0:
            self.target_net.load_state_dict(self.net.state_dict())

        log = {'total_reward': torch.tensor(self.total_reward).to(device),
               'reward': torch.tensor(reward).to(device),
               'train_loss': loss
               }
        status = {'steps': torch.tensor(self.global_step).to(device),
                  'total_reward': torch.tensor(self.total_reward).to(device)}

        return OrderedDict({'loss': loss, 'log': log, 'progress_bar': status})

    def configure_optimizers(self) -> List[Optimizer]:
        """ Initialize Adam optimizer"""
        optimizer = optim.Adam(self.net.parameters(), lr=self.training_parameters['learning_rate'])
        return [optimizer]

    def __dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences"""
        dataset = RLDataset(self.buffer, self.training_parameters['max_steps'])
        dataloader = DataLoader(dataset=dataset,
                                batch_size=self.training_parameters['batch_size'])
        return dataloader

    def train_dataloader(self) -> DataLoader:
        """Get train loader"""
        return self.__dataloader()

    def get_device(self, batch) -> str:
        """Retrieve device currently being used by minibatch"""
        return batch[0]['current_room'].device.index if self.on_gpu else 'cpu'


def main(mwg, model_parameters, training_parameters) -> None:
    model = DQNLightning(mwg, model_parameters, training_parameters)

    trainer = pl.Trainer(
        gpus=1,
        distributed_backend='dp',
        max_epochs=500,
        early_stop_callback=False,
        val_check_interval=100)
        # precision=16)
        # auto_lr_find=True
    # trainer.tune(model)
    trainer.fit(model)
