from agents import RandomBaseline
from agents.rl_baseline import Net
from MapWorld import MapWorldGym
from utils import evaluation
import numpy as np
import time
from tqdm import tqdm

import torch
from torch import optim
from transformers import AutoTokenizer, BertModel


def discount_rewards(rewards, gamma=0.99):
    r = np.array([gamma**i * rewards[i]
        for i in range(len(rewards))])
    # Reverse the array direction for cumsum and then
    # revert back to the original order
    r = r[::-1].cumsum()[::-1]
    return r - r.mean()


if __name__ == '__main__':
    mwg = MapWorldGym()
    # mwg.reset()
    available_actions = mwg.total_available_actions
    action_space = np.arange(len(available_actions))
    # print(available_actions)
    # print(len(available_actions))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
    embeddings = BertModel.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

    max_sequence_length = 50

    lr = 0.01 # learning rate

    ntokens = len(tokenizer.get_vocab())  # the size of vocabulary
    emsize = 200  # embedding dimension
    nhead = 2  # the number of heads in the multiheadattention models
    nhid = 200  # the dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 2  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    dropout = 0.2  # the dropout value
    model = Net(ntokens, emsize, nhead, nhid, nlayers, dropout, max_sequence_length, len(available_actions)).to(device)
    # TODO maybe also use lr scheduler (adjust lr if) // Gradient clipping
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    total_rewards = []
    batch_rewards = []
    batch_actions = []
    batch_states_image = []
    batch_states_text = []
    batch_counter = 1

    num_episodes = 20
    batch_size = 4
    gamma = 0.99

    max_steps = 10

    for i in range(num_episodes):
        s_0 = mwg.reset()
        text = s_0[1]

        states_image = []
        states_text = []
        rewards = []
        actions = []

        done = False
        steps = 0
        # loop until an episode is finished or 30 steps have been done
        # 30 steps is worse than a agent with random actions
        print('\n')
        while not done and steps < max_steps:
            # print(steps)
            # preprocess state (image and text)
            im = s_0[0]
            im = np.reshape(im, (np.shape(im)[2], np.shape(im)[1], np.shape(im)[0]))
            im_tensor = torch.FloatTensor([im]).to(device)

            # TODO atm only supports feeding directions because of memory constraints
            #   max sequence length of the transformer 512 token
            text = s_0[2] #text + ' ' + s_0[2]
            inputs = tokenizer(text, padding='max_length', max_length=max_sequence_length)
            inputs = inputs['input_ids']
            inputs_tensor = torch.LongTensor([inputs]).to(device)
            src_mask = model.generate_square_subsequent_mask(inputs_tensor.size(0)).to(device)

            # print(text)
            # print('image tensor', a_im.size(), type(a_im))
            # print('text tensor:', a_inputs.size(), type(a_inputs))

            action_probabilities = model(im_tensor, inputs_tensor, src_mask)
            action_probabilities = action_probabilities.cpu().detach().numpy()[0]
            # print('model output', f, f.size(), type(f))
            # TODO change action selection to np.random.choice
            action = np.random.choice(action_space, p=action_probabilities)
            # print('action: ', action, available_actions[action])

            s_1, reward, done, _ = mwg.step(action)

            states_image.append(im)
            states_text.append(inputs)
            rewards.append(reward)
            actions.append(action)
            s_0 = s_1
            steps += 1

        # when a episode is finished collect experience
        batch_rewards.extend(discount_rewards(
            rewards, gamma))
        batch_states_image.extend(states_image)
        batch_states_text.extend(states_text)
        batch_actions.extend(actions)
        batch_counter += 1
        total_rewards.append(mwg.model_return)

        # print(np.shape(states_image), np.shape(states_text), np.shape(rewards), np.shape(actions))
        # print('batches shape', np.shape(batch_states_image), np.shape(batch_states_text), np.shape(batch_rewards), np.shape(batch_actions))
        # print('total model steps', mwg.model_steps)

        if batch_counter == batch_size:
            # optimizer.zero_grad()
            print('Training')
            im_tensor = torch.FloatTensor(batch_states_image).to(device)
            inputs_tensor = torch.LongTensor(batch_states_text).to(device)
            src_mask = model.generate_square_subsequent_mask(inputs_tensor.size(0)).to(device)

            reward_tensor = torch.FloatTensor(batch_rewards).to(device)
            action_tensor = torch.LongTensor(batch_actions).to(device)
            # print('actions', type(batch_actions), np.shape(batch_actions), type(action_tensor), np.shape(action_tensor))
            print(batch_actions)
            print(action_tensor)
            # calculate loss
            # print('model ', model(im_tensor, inputs_tensor, src_mask))
            logprob = torch.log(model(im_tensor, inputs_tensor, src_mask))

            selected_logprobs = reward_tensor * \
                                torch.gather(logprob, 1, action_tensor.unsqueeze(1)).squeeze()
            # print('selected_logprobs', selected_logprobs)
            loss = -selected_logprobs.mean()
            print('loss', loss)
            # Calculate gradients

            loss.backward()
            # Apply gradients
            optimizer.step()

            batch_rewards = []
            batch_actions = []
            batch_states = []
            batch_counter = 1
