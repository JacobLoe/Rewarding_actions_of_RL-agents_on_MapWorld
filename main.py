from agents import RandomBaseline
from agents.rl_baseline import Net
from MapWorld import MapWorldGym
from utils import evaluation
import numpy as np
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import torchtext
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

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

    # net = Net()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # batch, channel, width, height
    i = initial_state[0]
    i = np.reshape(i, (np.shape(i)[2], np.shape(i)[1], np.shape(i)[0]))
    x = torch.FloatTensor([i, i]).to(device)

    tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

    text = initial_state[1] + ' ' + initial_state[2]

    inputs = tokenizer.encode_plus(text)

    ntokens = len(tokenizer.get_vocab())  # the size of vocabulary
    emsize = 200  # embedding dimension
    nhid = 200  # the dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 2  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 2  # the number of heads in the multiheadattention models
    dropout = 0.2  # the dropout value
    print('ntokens', ntokens)
    model = Net(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)

    a = inputs['input_ids'] #torch.tensor(inputs['input_ids'])

    # print(a)
    # print(type(a), a.size())

    print('\n')

    c = [a, a]

    c = torch.LongTensor(c).to(device)

    print('text tensor:', c, c.size(), type(c))

    src_mask = model.generate_square_subsequent_mask(c.size(0)).to(device)

    f = model(x, c, src_mask) #torch.tensor(inputs['input_ids'], device=device), torch.tensor(inputs['attention_mask'], device=device))
    print('output', f, f.size(), type(f))

