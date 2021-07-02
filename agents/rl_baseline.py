import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math
from transformers import AutoTokenizer, BertModel
from tqdm import tqdm


# adapted from:
# https://towardsdatascience.com/learning-reinforcement-learning-reinforce-with-pytorch-5e8ad7fc7da0
def reinforce(mwg):
    # TODO make model weights save-/loadable
    available_actions = mwg.total_available_actions
    action_space = np.arange(len(available_actions))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    bert_embeddings = BertModel.from_pretrained('bert-base-uncased')

    lr = 0.01 # learning rate

    emsize = 768  # embedding size of the bert model
    nhead = 2  # the number of heads in the multiheadattention models
    nhid = 200  # the dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 2  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    dropout = 0.2  # the dropout value
    max_sequence_length = 25    # maximum length the text state of the env will get padded to
    model = RLBaseline(emsize, nhead, nhid, nlayers, dropout, max_sequence_length, len(available_actions)).to(device)
    # TODO maybe also use lr scheduler (adjust lr if) // Gradient clipping
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    total_rewards = []
    total_steps = []

    batch_rewards = []
    batch_actions = []
    batch_states_image = []
    batch_states_text = []
    batch_counter = 2

    num_episodes = 20
    batch_size = 1
    gamma = 0.99

    max_steps = 10

    for ep in tqdm(range(num_episodes)):

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
        while not done and steps < max_steps:
            # preprocess state (image and text)
            im = s_0[0]
            im = np.reshape(im, (np.shape(im)[2], np.shape(im)[1], np.shape(im)[0]))
            im_tensor = torch.FloatTensor([im])

            # TODO atm only supports feeding directions because of memory constraints
            #   max sequence length of the transformer 512 token
            text = s_0[1] + ' ' + s_0[2] #text + ' ' + s_0[2]

            text_tokens = tokenizer.encode(text, padding='max_length', max_length=max_sequence_length)
            text_tokens_tensor = torch.LongTensor(text_tokens).unsqueeze(0)

            bert_output = bert_embeddings(text_tokens_tensor)
            embedded_text = bert_output[0][0]
            embedded_text = embedded_text.cpu().detach().numpy()

            embedded_text_tensor = torch.LongTensor([embedded_text])
            src_mask = model.generate_square_subsequent_mask(embedded_text_tensor.size(0))

            action_probabilities = model(im_tensor.to(device),
                                         embedded_text_tensor.to(device),
                                         src_mask.to(device))

            action_probabilities = action_probabilities.cpu().detach().numpy()[0]
            action = np.random.choice(action_space, p=action_probabilities)

            s_1, reward, done, _ = mwg.step(action)

            states_image.append(im)
            states_text.append(embedded_text)
            rewards.append(reward)
            actions.append(action)

            s_0 = s_1
            steps += 1

            if done or steps >= max_steps:
                # when a episode is finished, collect experience
                batch_rewards.extend(discount_rewards(
                    rewards, gamma))
                batch_states_image.extend(states_image)
                batch_states_text.extend(states_text)
                batch_actions.extend(actions)
                batch_counter += 1
                total_rewards.append(mwg.model_return)
                total_steps.append(steps)

                if batch_counter == batch_size:
                    optimizer.zero_grad()

                    # cast the batch to tensors and onto the GPU
                    im_tensor = torch.FloatTensor(batch_states_image).to(device)
                    inputs_tensor = torch.LongTensor(batch_states_text).to(device)
                    src_mask = model.generate_square_subsequent_mask(inputs_tensor.size(0)).to(device)
                    reward_tensor = torch.FloatTensor(batch_rewards).to(device)
                    action_tensor = torch.LongTensor(batch_actions).to(device)

                    #
                    logprob = torch.log(model(im_tensor, inputs_tensor, src_mask))
                    selected_logprobs = reward_tensor * \
                                        torch.gather(logprob, 1, action_tensor.unsqueeze(1)).squeeze()
                    loss = -selected_logprobs.mean()

                    # Calculate gradients
                    loss.backward()
                    # Apply gradients
                    optimizer.step()

                    batch_rewards = []
                    batch_actions = []
                    batch_states_image = []
                    batch_states_text = []
                    batch_counter = 1
                avg_rewards = np.mean(total_rewards[-100:])
                # Print running average
                print("\rEp: {} Average of last 100: {:.2f}".format(ep + 1, avg_rewards), end="")

    return total_rewards, total_steps


def discount_rewards(rewards, gamma=0.99):
    r = np.array([gamma**i * rewards[i]
        for i in range(len(rewards))])
    # Reverse the array direction for cumsum and then
    # revert back to the original order
    r = r[::-1].cumsum()[::-1]
    return r - r.mean()


class RLBaseline(nn.Module):
    def __init__(self, emsize, nhead, nhid, nlayers, dropout, max_sequence_length, output_size):
        super(RLBaseline, self).__init__()

        # CNN
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(393120, 1200)   # layer size is result of image res (480x854) after conv + pool
        self.fc2 = nn.Linear(1200, 840)
        self.fc3 = nn.Linear(840, max_sequence_length)

        # Transformer
        # adapted from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        self.pos_encoder = PositionalEncoding(emsize, dropout)
        encoder_layers = TransformerEncoderLayer(emsize, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        self.fc4 = nn.Linear(2*max_sequence_length, output_size)

        # self.init_weights()

    def forward(self, im, src, src_mask):
        x = self.pool(F.relu(self.conv1(im)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)     # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = output.mean(dim=2)

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
        # self.encoder.weight.data.uniform_(-initrange, initrange)
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
