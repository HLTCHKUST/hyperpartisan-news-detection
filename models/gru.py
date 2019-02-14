import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

class GRU(nn.Module):
    def __init__(self, args, embedding_matrix=None):
        super(GRU, self).__init__()
        self.args = args
        self.embedding_matrix = embedding_matrix
        self.batch_size = self.args.batch_size

        self.V = args.vocab_size
        self.D = args.embed_dim
        self.H = args.hidden_dim
        self.L = args.layer_num
        self.C = args.class_num
        
        self.num_directions = 1

        # Initialize Embedding lookup
        self.embed = nn.Embedding(self.V, self.D)
        if self.args.load_embeddings:
            self.embed.weight = nn.Parameter(self.embedding_matrix)

        if self.args.cuda:
            self.embed = self.embed.cuda()
        else:
            self.embed = self.embed.cpu()

        if self.args.static:
            self.embed.weight.requires_grad = False

        self.hidden = self.init_hidden()

        self.gru = nn.GRU(input_size=self.D,
                          hidden_size=self.H,
                          num_layers=self.L,
                          dropout=self.args.dropout,
                          bidirectional=self.args.bidirectional)

        self.dropout = nn.Dropout(args.dropout)

        self.total_hidden_size = self.H * self.L

        if self.args.bidirectional:
            self.total_hidden_size *= 2
            self.num_directions = 2

        self.output_size = self.total_hidden_size
        self.out = nn.Linear(self.output_size, self.C)

    def init_hidden(self):
        h0 = torch.zeros(self.L * self.num_directions, self.batch_size, self.H)
        if self.args.cuda:
            h0 = Variable(h0.cuda())
        else:
            h0 = Variable(h0)
        return h0

    # sequence: (seq_len, batch_size)
    def forward(self, sequence, seq_lens):
        # dim: (max_seq_len, batch_size, D)
        x = self.embed(sequence)
        x = self.dropout(x)

        x = nn.utils.rnn.pack_padded_sequence(x, seq_lens)

        # gru_input: (-1, batch_size, D)
        # gru_output: (seq_len, batch_size, H) # all the hidden states for each timesteps
        # hidden: last hidden state
        gru_out, self.hidden = self.gru(x, self.hidden)

        if self.use_attn:
            gru_out, _ = nn.utils.rnn.pad_packed_sequence(gru_out,
                                                          batch_first=True)
        else:
            # gru_out, _ = nn.utils.rnn.pad_packed_sequence(gru_out)
            # only need the last hidden state without attention
            x = self.hidden.squeeze()
            # x = gru_out[-1].squeeze()

            # concatenate into single vector
            if self.args.bidirectional:
                x = torch.cat((self.hidden[0], self.hidden[1]), dim=1)

        # Dropout
        x = self.dropout(x)
        logit = self.out(x)
        
        return (x, logit) if is_adversarial else logit

    def predict(self, x):
        # logit = self.forward(x)
        pass

    def predict_proba(self, x, seq_lens):
        return F.sigmoid(self.forward(x, seq_lens))
