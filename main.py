from agents import RandomBaseline
from agents.rl_baseline import Net
from MapWorld import MapWorldGym
from utils import evaluation
import numpy as np
import time

import torch
from transformers import AutoTokenizer

if __name__ == '__main__':
    mwg = MapWorldGym()
    initial_state = mwg.reset()
    available_actions = mwg.total_available_actions
    # print(initial_state)

    # rb = RandomBaseline()
    #
    # model_return, model_steps = evaluation.evaluate_model(mwg, rb, evaluation.eval_rand_baseline, num_iterations=5)
    # print('\n-------------------')
    # print('Return per model run: ', model_return)
    # print('Mean return: ', np.mean(model_return))
    # print('-------------------')
    # print('Total steps per model run', model_steps)
    # print('Mean steps: ', np.mean(model_steps))
    #
    # evaluation.create_histograms(model_return, model_steps)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

    ntokens = len(tokenizer.get_vocab())  # the size of vocabulary
    emsize = 200  # embedding dimension
    nhead = 2  # the number of heads in the multiheadattention models
    nhid = 200  # the dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 2  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    dropout = 0.2  # the dropout value
    model = Net(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)

    total_rewards = []
    batch_rewards = []
    batch_actions = []
    batch_states = []

    #####
    # episode 1, step 1

    # batch, channel, width, height
    i = initial_state[0]
    i = np.reshape(i, (np.shape(i)[2], np.shape(i)[1], np.shape(i)[0]))

    text = initial_state[1] + ' ' + initial_state[2]
    inputs = tokenizer.encode_plus(text)
    inputs = inputs['input_ids']

    # im = [i]
    #
    # t = [inputs]

    x = torch.FloatTensor([i]).to(device)
    c = torch.LongTensor([inputs]).to(device)
    src_mask = model.generate_square_subsequent_mask(c.size(0)).to(device)

    print(text)
    print('text tensor:', c.size(), type(c))
    print('image tensor', x.size(), type(x))

    f = model(x, c, src_mask)
    print('output', f, f.size(), type(f))

    ai = int(torch.argmax(f).cpu().detach())
    action = available_actions[ai]
    print('action: ', action)

    state, reward, done, _ = mwg.step(action)

    states_image = []
    states_text = []
    rewards = []
    actions = []


    #####
    # episode 1, step 2
    print('\n')

    i = state[0]
    i = np.reshape(i, (np.shape(i)[2], np.shape(i)[1], np.shape(i)[0]))

    text = text + ' ' + state[1] + ' ' + state[2]
    inputs = tokenizer.encode_plus(text)
    inputs = inputs['input_ids']

    states_image.append(i)
    states_text.append(inputs)
    rewards.append(reward)
    actions.append(action)

    print(np.shape(states_image), np.shape(states_text), np.shape(rewards), np.shape(actions))

    x = torch.FloatTensor([i]).to(device)
    c = torch.LongTensor([inputs]).to(device)
    src_mask = model.generate_square_subsequent_mask(c.size(0)).to(device)

    print(text)
    print('text tensor:', c.size(), type(c))
    print('image tensor', x.size(), type(x))

    f = model(x, c, src_mask)

    print('output', f, f.size(), type(f))

    ai = int(torch.argmax(f).cpu().detach())
    action = available_actions[ai]
    print('action: ', action)

    state, reward, done, _ = mwg.step(action)

    #####
    # episode 1, step 3
    print('\n')

    i = state[0]
    i = np.reshape(i, (np.shape(i)[2], np.shape(i)[1], np.shape(i)[0]))

    text = text + ' ' + state[1] + ' ' + state[2]
    inputs = tokenizer.encode_plus(text)
    inputs = inputs['input_ids']

    states_image.append(i)
    states_text.append(inputs)
    rewards.append(reward)
    actions.append(action)

    print(np.shape(states_image), np.shape(states_text), np.shape(rewards), np.shape(actions))

    x = torch.FloatTensor([i]).to(device)
    c = torch.LongTensor([inputs]).to(device)
    src_mask = model.generate_square_subsequent_mask(c.size(0)).to(device)

    print(text)
    print('text tensor:', c.size(), type(c))
    print('image tensor', x.size(), type(x))

    f = model(x, c, src_mask)

    print('output', f, f.size(), type(f))

    ai = int(torch.argmax(f).cpu().detach())
    action = available_actions[ai]
    print('action: ', action)

    state, reward, done, _ = mwg.step(action)

    print('\n')

    # im.append(i)
    # t.append(inputs)

    # logprob = torch.log(
    #     model(x,c, src_mask))
    # selected_logprobs = reward_tensor * \
    #                     torch.gather(logprob, 1,
    #                                  action_tensor.unsqueeze(1)).squeeze()
    # loss = -selected_logprobs.mean()