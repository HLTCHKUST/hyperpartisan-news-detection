#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from utils.data_reader_hier import prepare_data
from utils import constant

def batch_matmul_bias(seq, weight, bias, nonlinearity=''):
    s = None
    bias_dim = bias.size()
    for i in range(seq.size(0)):
        _s = torch.mm(seq[i], weight)
        _s = torch.add(_s, bias.expand(_s.size()[0], bias_dim[0]))
        if(nonlinearity=='tanh'):
            _s = torch.tanh(_s)
        _s = _s.unsqueeze(0)
        if(s is None):
            s = _s
        else:
            s = torch.cat((s,_s),0)
    return s.squeeze()

def batch_matmul(seq, weight, nonlinearity=''):
    s = None
    for i in range(seq.size()[0]):
        _s = torch.mm(seq[i], weight)
        if(nonlinearity=='tanh'):
            _s = torch.tanh(_s)
        _s = _s.unsqueeze(0)
        if(s is None):
            s = _s
        else:
            s = torch.cat((s,_s),0)
    return s.squeeze()

def attention_mul(rnn_outputs, att_weights):
    attn_vectors = None
    for i in range(rnn_outputs.size(0)):
        h_i = rnn_outputs[i]
        a_i = att_weights[i].unsqueeze(1).expand_as(h_i)
        h_i = torch.mul(h_i, a_i)
        h_i = h_i.unsqueeze(0)
        if(attn_vectors is None):
            attn_vectors = h_i
        else:
            attn_vectors = torch.cat((attn_vectors, h_i),0)
    return torch.sum(attn_vectors, 0).unsqueeze(0)

# ## Word attention model with bias
class AttentionWordRNN(nn.Module):
    def __init__(self, batch_size, num_vocab, embed_size, word_gru_hidden, bidirectional= True):        
        super(AttentionWordRNN, self).__init__()
        
        self.batch_size = batch_size
        self.num_vocab = num_vocab
        self.embed_size = embed_size
        self.word_gru_hidden = word_gru_hidden
        self.bidirectional = bidirectional

        self.lookup = nn.Embedding(num_vocab, embed_size)
        if bidirectional == True:
            self.word_gru = nn.GRU(embed_size, word_gru_hidden, bidirectional=True)
            self.weight_W_word = nn.Parameter(torch.Tensor(2* word_gru_hidden,2*word_gru_hidden))
            self.bias_word = nn.Parameter(torch.Tensor(2* word_gru_hidden))
            self.weight_proj_word = nn.Parameter(torch.Tensor(2*word_gru_hidden, 1))
        else:
            self.word_gru = nn.GRU(embed_size, word_gru_hidden, bidirectional=False)
            self.weight_W_word = nn.Parameter(torch.Tensor(word_gru_hidden, word_gru_hidden))
            self.bias_word = nn.Parameter(torch.Tensor(word_gru_hidden))
            self.weight_proj_word = nn.Parameter(torch.Tensor(word_gru_hidden, 1))
        
        self.softmax_word = nn.Softmax(dim=1)
        self.weight_W_word.data.uniform_(-0.01, 0.01)
        self.bias_word.data.uniform_(-0.01, 0.01)
        self.weight_proj_word.data.uniform_(-0.01, 0.01)

    def forward(self, embed):
        # embeddings
        embed = self.lookup(embed)
        # print("embedded:", embedded.size())
        # word level gru
        output_word, _ = self.word_gru(embed)
        word_squish = batch_matmul_bias(output_word, self.weight_W_word, self.bias_word, nonlinearity='tanh')
        word_attn = batch_matmul(word_squish, self.weight_proj_word)
        word_attn_vectors = attention_mul(output_word, self.softmax_word(word_attn.transpose(1,0)).transpose(1,0))
        # word_attn_vectors = attention_mul(output_word, word_attn_norm.transpose(1,0))
        return word_attn_vectors
    
    def init_hidden(self):
        if self.bidirectional == True:
            return Variable(torch.zeros(2, self.batch_size, self.word_gru_hidden))
        else:
            return Variable(torch.zeros(1, self.batch_size, self.word_gru_hidden))        


# ## Sentence Attention model with bias
class AttentionSentRNN(nn.Module):
    
    def __init__(self, batch_size, sent_gru_hidden, word_gru_hidden, n_classes, bidirectional= True):        
        super(AttentionSentRNN, self).__init__()
        
        self.batch_size = batch_size
        self.sent_gru_hidden = sent_gru_hidden
        self.n_classes = n_classes
        self.word_gru_hidden = word_gru_hidden
        self.bidirectional = bidirectional
        
        if bidirectional == True:
            self.sent_gru = nn.GRU(2 * word_gru_hidden, sent_gru_hidden, bidirectional=True)        
            self.weight_W_sent = nn.Parameter(torch.Tensor(2* sent_gru_hidden ,2* sent_gru_hidden))
            self.bias_sent = nn.Parameter(torch.Tensor(2* sent_gru_hidden))
            self.weight_proj_sent = nn.Parameter(torch.Tensor(2* sent_gru_hidden, 1))
            # self.final_linear = nn.Linear(2* sent_gru_hidden, n_classes)
        else:
            self.sent_gru = nn.GRU(word_gru_hidden, sent_gru_hidden, bidirectional=True)        
            self.weight_W_sent = nn.Parameter(torch.Tensor(sent_gru_hidden ,sent_gru_hidden))
            self.bias_sent = nn.Parameter(torch.Tensor(sent_gru_hidden))
            self.weight_proj_sent = nn.Parameter(torch.Tensor(sent_gru_hidden, 1))
            # self.final_linear = nn.Linear(sent_gru_hidden, n_classes)
        self.softmax_sent = nn.Softmax(dim=1)
        self.weight_W_sent.data.uniform_(-0.01, 0.01)
        self.bias_sent.data.uniform_(-0.01, 0.01)
        self.weight_proj_sent.data.uniform_(-0.01, 0.01)
        
    def forward(self, word_attention_vectors):
        # print(word_attention_vectors.size())
        output_sent, _ = self.sent_gru(word_attention_vectors)
        sent_squish = batch_matmul_bias(output_sent, self.weight_W_sent, self.bias_sent, nonlinearity='tanh')
        sent_attn = batch_matmul(sent_squish, self.weight_proj_sent)
        sent_attn_vectors = attention_mul(output_sent, self.softmax_sent(sent_attn.transpose(1,0)).transpose(1,0))
        # sent_attn_vectors = attention_mul(output_sent, sent_attn_norm.transpose(1,0))
        sent_attn_vectors = sent_attn_vectors.squeeze(0)
        # final classifier
        # final_map = self.final_linear(sent_attn_vectors)
        return sent_attn_vectors
    
    def init_hidden(self):
        if self.bidirectional == True:
            return Variable(torch.zeros(2, self.batch_size, self.sent_gru_hidden))
        else:
            return Variable(torch.zeros(1, self.batch_size, self.sent_gru_hidden))   

# hierarchical attention model
class HierarchicalAttenModel(nn.Module):
    
    def __init__(self, word_attn_model, sent_attn_model):
        super(HierarchicalAttenModel, self).__init__()
        self.word_attn_model = word_attn_model
        self.sent_attn_model = sent_attn_model
    
    def forward(self, X):
        # X.size() --> (batch_size, max_sents, max_tokens)
        batch_size, max_sents, max_tokens = X.size()
        s = None
        for i in range(max_sents):
            # print(state_word)
            _s = self.word_attn_model(X[:,i,:].transpose(1,0))
            if s is None:
                s = _s
            else:
                s = torch.cat((s, _s), 0)
        # print(s.size())
        sent_attn_vectors = self.sent_attn_model(s)

        return sent_attn_vectors

# # ## Functions to train the model
# def train_data(mini_batch, targets, word_attn_model, sent_attn_model, word_optimizer, sent_optimizer, criterion):
#     state_word = word_attn_model.init_hidden().cuda()
#     state_sent = sent_attn_model.init_hidden().cuda()
#     max_sents, batch_size, max_tokens = mini_batch.size()
#     word_optimizer.zero_grad()
#     sent_optimizer.zero_grad()
#     s = None
#     for i in range(max_sents):
#         _s, state_word, _ = word_attn_model(mini_batch[i,:,:].transpose(0,1), state_word)
#         if(s is None):
#             s = _s
#         else:
#             s = torch.cat((s,_s),0)            
#     y_pred, state_sent, _ = sent_attn_model(s, state_sent)
#     loss = criterion(y_pred.cuda(), targets) 
#     loss.backward()

#     word_optimizer.step()
#     sent_optimizer.step()
    
#     return loss.data[0]

def get_predictions(val_tokens, word_attn_model, sent_attn_model):
    max_sents, batch_size, max_tokens = val_tokens.size()
    state_word = word_attn_model.init_hidden().cuda()
    state_sent = sent_attn_model.init_hidden().cuda()
    s = None
    for i in range(max_sents):
        _s, state_word, _ = word_attn_model(val_tokens[i,:,:].transpose(0,1), state_word)
        if(s is None):
            s = _s
        else:
            s = torch.cat((s,_s),0)            
    y_pred, state_sent, _ = sent_attn_model(s, state_sent)    
    return y_pred

# word_attn = AttentionWordRNN(batch_size=64, num_vocab=133253, embed_size=300, 
#                              word_gru_hidden=100, bidirectional= True)

# sent_attn = AttentionSentRNN(batch_size=64, sent_gru_hidden=100, word_gru_hidden=100, 
#                              n_classes=10, bidirectional= True)

# learning_rate = 1e-1
# momentum = 0.9
# word_optmizer = torch.optim.SGD(word_attn.parameters(), lr=learning_rate, momentum= momentum)
# sent_optimizer = torch.optim.SGD(sent_attn.parameters(), lr=learning_rate, momentum= momentum)
# criterion = nn.NLLLoss()

# word_attn.cuda()
# sent_attn.cuda()

# import pickle
# x_save_path = "data_new/hier_X_{}_full_v2.pickle"
# y_save_path = "data_new/hier_y_{}_full_v2.pickle"
# # x_save_path = "data_new/hier_X_{}_smallest.pickle"
# # y_save_path = "data_new/hier_y_{}_smallest.pickle"

# def prepare_fakenews_data():
#     with open(x_save_path.format('train'), 'rb') as handle:
#         X_train = pickle.load(handle)
#     with open(x_save_path.format('val'), 'rb') as handle:
#         X_test = pickle.load(handle)
    
#     with open(y_save_path.format('train'), 'rb') as handle:
#         y_train = pickle.load(handle)
#     with open(y_save_path.format('val'), 'rb') as handle:
#         y_test = pickle.load(handle)
    
#     return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)

# X_train_all, X_test, y_train_all, y_test = prepare_fakenews_data()


# from sklearn.model_selection import train_test_split
# X_train, X_val, y_train, y_val = train_test_split(X_train_all, y_train_all, test_size = 0.2, random_state= 42)

# def pad_batch(mini_batch):
# #     print(len(mini_batch))
#     mini_batch_size = len(mini_batch)
#     max_sent_len = int(np.mean([len(x) for x in mini_batch]))
#     max_token_len = int(np.mean([len(val) for sublist in mini_batch for val in sublist]))
#     main_matrix = np.zeros((mini_batch_size, max_sent_len, max_token_len), dtype= np.int)
#     for i in range(main_matrix.shape[0]):
#         for j in range(main_matrix.shape[1]):
#             for k in range(main_matrix.shape[2]):
#                 try:
#                     main_matrix[i,j,k] = mini_batch[i][j][k]
#                 except IndexError:
#                     pass
#     return Variable(torch.from_numpy(main_matrix).transpose(0,1))

# def test_accuracy_mini_batch(tokens, labels, word_attn, sent_attn):
#     y_pred = get_predictions(tokens, word_attn, sent_attn)
#     _, y_pred = torch.max(y_pred, 1)
#     correct = np.ndarray.flatten(y_pred.data.cpu().numpy())
#     labels = np.ndarray.flatten(labels.data.cpu().numpy())
#     num_correct = sum(correct == labels)
#     return float(num_correct) / len(correct)


# def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
#     assert len(inputs) == len(targets)
# #     assert inputs.shape[0] == targets.shape[0]
    
#     if shuffle:
#         indices = np.arange(len(inputs))
#         np.random.shuffle(indices)
#     for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
#         if shuffle:
#             excerpt = indices[start_idx:start_idx + batchsize]
#         else:
#             excerpt = slice(start_idx, start_idx + batchsize)
            
#         yield inputs[(excerpt)], targets[(excerpt)]

# def gen_minibatch(tokens, labels, mini_batch_size, shuffle= False): # True
#     for token, label in iterate_minibatches(tokens, labels, mini_batch_size, shuffle= shuffle):
#         token = pad_batch(token)
#         yield token.cuda(), Variable(torch.from_numpy(label), requires_grad= False).cuda()    

# def test_accuracy_full_batch(tokens, labels, mini_batch_size, word_attn, sent_attn):
#     p = []
#     l = []
#     g = gen_minibatch(tokens, labels, mini_batch_size)
#     for token, label in g:
#         y_pred = get_predictions(token.cuda(), word_attn, sent_attn)
#         _, y_pred = torch.max(y_pred, 1)
#         p.append(np.ndarray.flatten(y_pred.data.cpu().numpy()))
#         l.append(np.ndarray.flatten(label.data.cpu().numpy()))
#     p = [item for sublist in p for item in sublist]
#     l = [item for sublist in l for item in sublist]
#     p = np.array(p)
#     l = np.array(l)
#     num_correct = sum(p == l)
#     return float(num_correct)/ len(p)

# def test_data(mini_batch, targets, word_attn_model, sent_attn_model):    
#     state_word = word_attn_model.init_hidden().cuda()
#     state_sent = sent_attn_model.init_hidden().cuda()
#     max_sents, batch_size, max_tokens = mini_batch.size()
#     s = None
#     for i in range(max_sents):
#         _s, state_word, _ = word_attn_model(mini_batch[i,:,:].transpose(0,1), state_word)
#         if(s is None):
#             s = _s
#         else:
#             s = torch.cat((s,_s),0)            
#     y_pred, state_sent,_ = sent_attn_model(s, state_sent)
#     loss = criterion(y_pred.cuda(), targets)     
#     return loss.data[0]

# def check_val_loss(val_tokens, val_labels, mini_batch_size, word_attn_model, sent_attn_model):
#     val_loss = []
#     for token, label in iterate_minibatches(val_tokens, val_labels, mini_batch_size, shuffle= True):
#         val_loss.append(test_data(pad_batch(token).cuda(), Variable(torch.from_numpy(label), requires_grad= False).cuda(), 
#                                   word_attn_model, sent_attn_model))
#     return np.mean(val_loss)

# import time
# import math

# def timeSince(since):
#     now = time.time()
#     s = now - since
#     m = math.floor(s / 60)
#     s -= m * 60
#     return '%dm %ds' % (m, s)

# # ## Training
# def train_early_stopping(mini_batch_size, X_train, y_train, X_test, y_test, word_attn_model, sent_attn_model, 
#                          word_attn_optimiser, sent_attn_optimiser, loss_criterion, num_epoch, 
#                          print_val_loss_every = 1000, print_loss_every = 50):
#     start = time.time()
#     loss_full = []
#     loss_epoch = []
#     accuracy_epoch = []
#     loss_smooth = []
#     accuracy_full = []
#     epoch_counter = 0

#     # ENTRY
#     g = gen_minibatch(X_train, y_train, mini_batch_size)
#     for i in range(1, num_epoch + 1):
#         try:
#             tokens, labels = next(g)
#             loss = train_data(tokens, labels, word_attn_model, sent_attn_model, word_attn_optimiser, sent_attn_optimiser, loss_criterion)
#             acc = test_accuracy_mini_batch(tokens, labels, word_attn_model, sent_attn_model)
#             accuracy_full.append(acc)
#             accuracy_epoch.append(acc)
#             loss_full.append(loss)
#             loss_epoch.append(loss)
#             # print loss every n passes
#             if i % print_loss_every == 0:
#                 print ('Loss at %d minibatches, %d epoch,(%s) is %f' %(i, epoch_counter, timeSince(start), np.mean(loss_epoch)))
#                 print ('Accuracy at %d minibatches is %f' % (i, np.mean(accuracy_epoch)))
#             # check validation loss every n passes
#             if i % print_val_loss_every == 0:
#                 val_loss = check_val_loss(X_test, y_test, mini_batch_size, word_attn_model, sent_attn_model)
#                 print ('Average training loss at this epoch..minibatch..%d..is %f' % (i, np.mean(loss_epoch)))
#                 print ('Validation loss after %d passes is %f' %(i, val_loss))
#                 if val_loss > np.mean(loss_full):
#                     print ('Validation loss is higher than training loss at %d is %f , stopping training!' % (i, val_loss))
#                     print ('Average training loss at %d is %f' % (i, np.mean(loss_full)))
#         except StopIteration:
#             epoch_counter += 1
#             print ('Reached %d epocs' % epoch_counter)
#             print ('i %d' % i)
#             g = gen_minibatch(X_train, y_train, mini_batch_size)
#             loss_epoch = []
#             accuracy_epoch = []
#     return loss_full

# loss_full= train_early_stopping(64, X_train, y_train, X_val, y_val, word_attn, sent_attn, word_optmizer, sent_optimizer, 
#                             criterion, 5000, 1000, 50)

# test_accuracy_full_batch(X_test, y_test, 64, word_attn, sent_attn)

# test_accuracy_full_batch(X_train, y_train, 64, word_attn, sent_attn)