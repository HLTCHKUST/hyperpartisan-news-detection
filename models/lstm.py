import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
from utils.feature_utils import create_embedding_matrix, gen_embeddings

import numpy as np
import math

class LSTM(nn.Module):
    """
    An LSTM model. 
    Inputs: 
        X: (batch_size, seq_len)
        X_lengths: (batch_size)
    Outputs: (batch_size, labels)
    """
    def __init__(self, vocab, embedding_size, hidden_size, num_layers, pretrain_emb, max_length=700, input_dropout=0.0, layer_dropout=0.0, is_bidirectional=False):
        super(LSTM, self).__init__()
        self.embedding_dim = embedding_size
        self.hidden_size = hidden_size
        self.input_dropout = nn.Dropout(input_dropout)
        self.layer_dropout = nn.Dropout(layer_dropout)
        self.emb = nn.Embedding(vocab.n_words, embedding_size, padding_idx=0)
        if pretrain_emb:
            print("load pre-trained embedding")
            if self.embedding_dim == 100:
                embedding_matrix = gen_embeddings(vocab, self.embedding_dim, "../glove/glove.6B.100d.txt")
            else:
                embedding_matrix = gen_embeddings(vocab, self.embedding_dim, "../glove/glove.42B.300d.txt")
            self.emb.weight.data.copy_(torch.FloatTensor(embedding_matrix))
            self.emb.weight.requires_grad = False
        self.lstm = nn.LSTM(embedding_size, hidden_size=hidden_size, num_layers=num_layers, dropout=layer_dropout,
                            bidirectional=is_bidirectional, batch_first=True)

#         self.W = nn.Linear(hidden_size*2 if is_bidirectional else hidden_size, 2) ## 2 class
#         self.softmax = nn.Softmax(dim=1)
        
        self.W = nn.Linear(hidden_size, 1) ## 2 class
        self.sigmoid = nn.Sigmoid()
        
        self.num_layers = num_layers
        self.is_bidirectional = is_bidirectional

    def forward(self, X, X_lengths):
        X = self.emb(X)
        X = self.input_dropout(X)
        # X = X.transpose(0, 1) # (len, batch_size, dim)
        # _, hidden = self.lstm(X)
        packed_input = torch.nn.utils.rnn.pack_padded_sequence(X, X_lengths, batch_first=True)
        _, hidden = self.lstm(packed_input)
#         # returns hidden state of all timesteps as well as hidden state at last timestep
#         # should take last non zero hidden state, not last timestamp (may have zeros), don't take output
#         # output, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=False)

        last_hidden = hidden[-1] # (num_direction, batch_size, dim)
        if self.is_bidirectional:
            last_hidden = torch.cat((last_hidden[0].squeeze(0), last_hidden[1].squeeze(0)), dim=1)
        else:
            last_hidden = last_hidden.squeeze(0)
            
        last_hidden = self.layer_dropout(last_hidden)

        a_hat = self.W(last_hidden) # (batch_size, 1)

        return a_hat, self.sigmoid(a_hat)
    
    def feature(self, X, X_lengths):
        """ get the feature from the lstm model, return the last hidden layer """
        X = self.emb(X)
        X = self.input_dropout(X)
        # packed_input = torch.nn.utils.rnn.pack_padded_sequence(X, X_lengths, batch_first=True)
        # _, hidden = self.lstm(packed_input)
        _, hidden = self.lstm(X)
        last_hidden = hidden[-1] # (num_direction, batch_size, dim)
        if self.is_bidirectional:
            last_hidden = torch.cat((last_hidden[0].squeeze(0), last_hidden[1].squeeze(0)), dim=1)
        else:
            # last_hidden = last_hidden.squeeze(0)
            # got the last layer
            last_hidden = last_hidden[-1]
        # print(last_hidden.size())
        last_hidden = self.layer_dropout(last_hidden)

        return last_hidden