
import torch
import torch.nn as nn
from utils import constant
from torch.autograd import Function

class Discriminator(nn.Module):
    """
    An Discriminator model. Used for adversarisal training
    """
    def __init__(self, dropout = 0.5, topic_num=20):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(constant.hidden_dim, topic_num)
        self.relu = nn.ReLU()
        self.layer_dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, X):
        reverse_f = ReverseLayerF.apply(X, 0.1)
        reverse_f = self.fc1(reverse_f)
        reverse_f = self.layer_dropout(reverse_f)
        reverse_f = self.softmax(reverse_f)
        return reverse_f

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # print ("before:")
        # print (grad_output)
        output = grad_output.neg() * ctx.alpha
        # print ("after:")
        # print (output)
        return output, None


class Classifier(nn.Module):
    """
    An Discriminator model. Used for adversarisal training
    """
    def __init__(self, hidden_dim1, hidden_dim2, dropout = 0.8):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(hidden_dim1 + hidden_dim2, 64)
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim1 + hidden_dim2)
        self.batch_norm2 = nn.BatchNorm1d(64)
        self.layer_dropout = nn.Dropout(p=dropout)
        self.sigmoid = nn.Sigmoid()
        # self.softmax = nn.Softmax(dim=1)
        
    def forward(self, X):
        X = self.batch_norm1(X)
        X = self.fc1(X)
        X = self.batch_norm2(X)
        X = self.layer_dropout(X)
        X = self.relu(X)
        X = self.fc2(X)
        # X = self.layer_dropout(X)
        X = self.sigmoid(X)
        return X

class Classifier2(nn.Module):
    def __init__(self, hidden_dim1, hidden_dim2, dropout = 0.8):
        super(Classifier2, self).__init__()
        self.fc1 = nn.Linear(hidden_dim1 + hidden_dim2, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim1 + hidden_dim2)
        self.batch_norm2 = nn.BatchNorm1d(256)
        self.batch_norm3 = nn.BatchNorm1d(64)
        self.layer_dropout = nn.Dropout(p=dropout)
        self.sigmoid = nn.Sigmoid()
        # self.softmax = nn.Softmax(dim=1)
        
    def forward(self, X):
        X = self.batch_norm1(X)
        X = self.fc1(X)
        X = self.batch_norm2(X)
        X = self.layer_dropout(X)
        X = self.relu(X)
        X = self.fc2(X)
        X = self.layer_dropout(X)
        X = self.relu(X)
        X = self.fc3(X)
        X = self.sigmoid(X)
        return X

class LR(nn.Module):
    """
    An Discriminator model. Used for adversarisal training
    """
    def __init__(self, hidden_dim1, hidden_dim2, dropout = 0.5):
        super(LR, self).__init__()
        
        if constant.use_emo2vec_feat:
            self.batch_norm = nn.BatchNorm1d(hidden_dim1 + hidden_dim2 + constant.hidden_emo_dim)
            self.fc = nn.Linear(hidden_dim1 + hidden_dim2 + constant.hidden_emo_dim, 1)
        elif constant.use_url:
            self.batch_norm = nn.BatchNorm1d(hidden_dim1 + hidden_dim2 + constant.hidden_url_vec)
            self.fc = nn.Linear(hidden_dim1 + hidden_dim2 + constant.hidden_url_vec, 1)
        else:
            self.batch_norm = nn.BatchNorm1d(hidden_dim1 + hidden_dim2)
            self.fc = nn.Linear(hidden_dim1 + hidden_dim2, 1)
        # self.layer_dropout = nn.Dropout(p=dropout)
        self.sigmoid = nn.Sigmoid()
    def forward(self, X):
        X = self.batch_norm(X)
        X = self.fc(X)
        # X = self.layer_dropout(X)
        X = self.sigmoid(X)
        return X
