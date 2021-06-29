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

    # batch, channel, width, height
    i = initial_state[0]
    i = np.reshape(i, (np.shape(i)[2], np.shape(i)[1], np.shape(i)[0]))

    tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

    text = initial_state[1] + ' ' + initial_state[2]

    inputs = tokenizer.encode_plus(text)

    ntokens = len(tokenizer.get_vocab())  # the size of vocabulary
    emsize = 200  # embedding dimension
    nhid = 200  # the dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 2  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 2  # the number of heads in the multiheadattention models
    dropout = 0.2  # the dropout value
    model = Net(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)

    inputs = inputs['input_ids']

    im = [i]

    t = [inputs]

    for z in range(4):
        print(np.shape(im), type(im))
        print(np.shape(t), type(t))

        x = torch.FloatTensor(im).to(device)

        c = torch.LongTensor(t).to(device)

        print('text tensor:', c.size(), type(c))
        print('image tensor', x.size(), type(x))

        src_mask = model.generate_square_subsequent_mask(c.size(0)).to(device)

        f = model(x, c, src_mask)
        print('output', f, f.size(), type(f))

        im.append(i)
        t.append(inputs)
        print('\n')

