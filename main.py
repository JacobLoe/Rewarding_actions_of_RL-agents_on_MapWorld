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
    i = np.expand_dims(i, axis=0)
    x = torch.tensor(i).float().to(device)
    # net.to(device)
    # print(net(x))

    tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

    text = r"""
    The room is kitchen with a refrigerator, a sink and a dishwasher. Plates and cups are on the desk. The walls of the kitchen are blue.
    """

    inputs = tokenizer.encode_plus(text, add_special_tokens=True)
    # print(inputs)
    #
    print(len(tokenizer.get_vocab()))
    # TEXT = torchtext.data.Field(tokenize=get_tokenizer("basic_english"),
    #                             init_token='<sos>',
    #                             eos_token='<eos>',
    #                             lower=True)
    # train_data, val_data, test_data = torchtext.datasets.WikiText2.splits(TEXT)
    #
    # # print(train_data)
    #
    # TEXT.build_vocab(train_data)

    ntokens = len(tokenizer.get_vocab())  # the size of vocabulary
    emsize = 200  # embedding dimension
    nhid = 200  # the dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 2  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 2  # the number of heads in the multiheadattention models
    dropout = 0.2  # the dropout value
    model = Net(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)

    # print(type(torch.tensor(inputs['input_ids'], device=device)), type(torch.tensor(inputs['attention_mask'], device=device)))
    # print(type(inputs))

    a = time.time()

    f = model.forward(im=x, src=torch.tensor(inputs['input_ids'], device=device),
                        src_mask=torch.tensor(inputs['attention_mask'], device=device))
    print('output',f)
    print('type',type(f))
    print('shape', np.shape(f))
    print(time.time()-a)