import os
import numpy as np
from tqdm import tqdm
from collections import namedtuple

import torch
import torch.nn as nn
from torch.nn import DataParallel
import torch.nn.functional as F
from torch.distributions import Categorical
from torchvision import transforms

from utils.distances import load_inception
from sentence_transformers import SentenceTransformer


# adapted from https://github.com/pytorch/examples/blob/master/reinforcement_learning/actor_critic.py
def actor_critic(mwg, model_parameters, training_parameters, base_path,
                 logger, save_model, gpu, load_model):
    """

    Args:
        mwg:
        model_parameters:
        training_parameters:
        base_path:
        logger:
        save_model:
        gpu:
        load_model:

    Returns:

    """
    SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

    device = torch.device(gpu if torch.cuda.is_available() else "cpu")
    available_actions = mwg.total_available_actions

    model = DataParallel(ActorCriticModel(model_parameters['embedding_size'],
                                          model_parameters['hidden_layer_size'],
                                          output_size=len(available_actions)).to(device))

    inception = load_inception(device)
    preprocess = transforms.Compose([
        transforms.Resize(299),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])

    lr = training_parameters['learning_rate']
    num_episodes = int(training_parameters['num_episodes'])
    batch_size = training_parameters['batch_size']
    gamma = training_parameters['gamma']
    max_steps = training_parameters['max_steps']
    checkpoint_frequency = training_parameters['checkpoint_frequency']  # how often should a checkpoint be created
    starting_episode = 0

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    em_model = SentenceTransformer(model_parameters['sentence_embedding_model'])
    em_model.max_seq_length = model_parameters['max_sequence_length']

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

            # preprocess state (image and text)
            processed_frame = preprocess(state['current_room']).unsqueeze(0).to(device)
            with torch.no_grad():
                im = inception(processed_frame).squeeze().cpu().detach().numpy()
            im_tensor = torch.FloatTensor([im])

            embeddings = em_model.encode(state['text_state'])
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

                    del returns
                    del value_losses
                    del policy_losses

                    batch_rewards = []
                    batch_actions = []
                    batch_counter = 0

        # save the progress of the training every checkpoint_frequency episodes
        if episode % checkpoint_frequency == 0 and save_model:
            torch.save({
                'current_episode': episode,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, ck_path)

    return total_rewards, total_steps, hits


class ActorCriticModel(nn.Module):
    def __init__(self, emsize, hidden_layer_size, output_size):
        super(ActorCriticModel, self).__init__()
        # TODO with the sentence transformer max_sequence_length is not a thing anymore
        # TODO maybe replace it with something else (emsize, etc)
        # TODO https://arxiv.org/pdf/1902.07742.pdf

        self.fc_image = nn.Linear(2048, 2*hidden_layer_size)

        self.fc_text = nn.Linear(emsize, 2*hidden_layer_size)

        # action model
        self.fc_action = nn.Linear(2*hidden_layer_size, hidden_layer_size)
        self.fc_action0 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.fc_action1 = nn.Linear(hidden_layer_size, output_size)

        # value model
        self.fc_value = nn.Linear(2*hidden_layer_size, hidden_layer_size)
        self.fc_value0 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.fc_value1 = nn.Linear(hidden_layer_size, 1)

    def forward(self, image, text):
        image = torch.tanh(self.fc_image(image))

        text = torch.tanh(self.fc_text(text))

        # combine image and text into one vector
        output = torch.mul(image, text)

        # compute the best action for a state
        actions = F.relu(self.fc_action(output))
        actions = F.relu(self.fc_action0(actions))
        actions = self.fc_action1(actions)
        actions = F.softmax(actions, dim=1)

        # compute the value of being in a state
        value = F.relu(self.fc_value(output))
        value = F.relu(self.fc_value0(value))
        value = self.fc_value1(value)

        return actions, value
