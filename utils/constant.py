import argparse
import random
import numpy as np
import torch
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("--hidden_dim", type=int, default=100)
parser.add_argument("--emb_dim", type=int, default=300)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--optimizer", type=str, default='adam')
parser.add_argument("--input_dropout", type=float, default=0)
parser.add_argument("--layer_dropout", type=float, default=0)
parser.add_argument("--attention_dropout", type=float, default=0)
parser.add_argument("--relu_dropout", type=float, default=0)
parser.add_argument("--save_path", type=str, default="save/")
parser.add_argument("--cuda", action="store_true")
parser.add_argument("--multiple_gpu", action="store_true")
parser.add_argument("--pretrain_emb", action="store_true")
parser.add_argument("--test", action="store_true")
parser.add_argument("--model", type=str, default="UTRS") # UTRS, LSTM
parser.add_argument("--weight_sharing", action="store_true")
parser.add_argument("--label_smoothing", action="store_true")
parser.add_argument("--noam", action="store_true")
parser.add_argument("--universal", action="store_true")
parser.add_argument("--act", action="store_true")
parser.add_argument("--act_loss_weight", type=float, default=0.001)
parser.add_argument("--seed", type=int, default=1234)
parser.add_argument("--debug_mode", action="store_true")
parser.add_argument("--build_vocab_flag", action="store_true")
parser.add_argument("--manual_name", type=str, default="LstmAr_LstmTit")

parser.add_argument("--use_topic_prob", action="store_true")
parser.add_argument("--use_publisher", action="store_true")
parser.add_argument("--use_topic_feat", action="store_true")
parser.add_argument("--use_emo2vec_feat", action="store_true")
parser.add_argument("--use_url", action="store_true")
parser.add_argument("--use_url_rnn", action="store_true")
parser.add_argument("--use_bert", action="store_true")
parser.add_argument("--use_bert_plus_lstm", action="store_true")
parser.add_argument("--train_cleaner_dataset", action="store_true")
parser.add_argument("--use_attn", action="store_true")
parser.add_argument("--reduce_input", action="store_true")
# for UTransformer
parser.add_argument("--use_utransformer", action="store_true")
parser.add_argument("--num_heads", type=int, default=2)
parser.add_argument("--key_value_depth", type=int, default=128)
parser.add_argument("--filter_size_article", type=int, default=128)
parser.add_argument("--filter_size_title", type=int, default=64)
parser.add_argument("--max_hops_article", type=int, default=6)
parser.add_argument("--max_hops_title", type=int, default=3)

parser.add_argument("--topic_num", type=int, default=20)
parser.add_argument("--publisher_num", type=int, default=158)
parser.add_argument("--weight_decay", type=float, default=0)

parser.add_argument("--num_split", type=int, default=10)
parser.add_argument("--max_epochs", type=int, default=10)

## lstm
parser.add_argument("--n_layers", type=int, default=1)

# learning rate
parser.add_argument("--lr_lstm", type=float, default=5e-4)
parser.add_argument("--lr_hier", type=float, default=5e-5)
parser.add_argument("--lr_title", type=float, default=5e-3)
parser.add_argument("--lr_discri", type=float, default=0.01)
parser.add_argument("--lr_classi", type=float, default=1e-3)

# for title dim
parser.add_argument("--hidden_dim_tit", type=int, default=50)
# for hier word hidden dim
parser.add_argument("--hier_dim_word", type=int, default=100)
# for hier sent hidden dim
parser.add_argument("--hier_dim_sent", type=int, default=100)

# choices
parser.add_argument("--do_train", action="store_true")
parser.add_argument("--do_predict", action="store_true")
parser.add_argument("--do_eval_bert_plus_lstm", action="store_true")
parser.add_argument("--do_cross_validation", action="store_true")
parser.add_argument("--bert_from_scratch", action="store_true")

parser.add_argument("--aug_count", type=str, default="")

arg = parser.parse_args()
print(arg)
model = arg.model

# choices
do_train = arg.do_train
do_predict = arg.do_predict
do_eval_bert_plus_lstm = arg.do_eval_bert_plus_lstm
do_cross_validation = arg.do_cross_validation
bert_from_scratch = arg.bert_from_scratch

# Hyperparameters
hidden_dim= arg.hidden_dim
hidden_dim_tit= arg.hidden_dim_tit
hidden_dim_url= 128
hidden_emo_dim = 100
hidden_url_vec = 30646
hier_dim_word = arg.hier_dim_word
hier_dim_sent = arg.hier_dim_sent
emb_dim= arg.emb_dim
batch_size= arg.batch_size
lr=arg.lr
lr_lstm = arg.lr_lstm
lr_hier = arg.lr_hier
lr_discri = arg.lr_discri
lr_classi = arg.lr_classi
lr_title = arg.lr_title
seed = arg.seed
num_split = arg.num_split
max_epochs = arg.max_epochs
optimizer = arg.optimizer
input_dropout = arg.input_dropout
layer_dropout = arg.layer_dropout
attention_dropout = arg.attention_dropout
relu_dropout = arg.relu_dropout
use_topic_prob = arg.use_topic_prob
use_publisher = arg.use_publisher
use_topic_feat = arg.use_topic_feat
use_emo2vec_feat = arg.use_emo2vec_feat
use_url = arg.use_url
use_url_rnn = arg.use_url_rnn
use_bert = arg.use_bert
use_bert_plus_lstm = arg.use_bert_plus_lstm
train_cleaner_dataset = arg.train_cleaner_dataset
use_attn = arg.use_attn
reduce_input = arg.reduce_input
# UTransformer
use_utransformer = arg.use_utransformer
num_heads = arg.num_heads
key_value_depth = arg.key_value_depth
filter_size_article = arg.filter_size_article
filter_size_title = arg.filter_size_title
max_hops_article = arg.max_hops_article
max_hops_title = arg.max_hops_title

topic_num = arg.topic_num
publisher_num = arg.publisher_num
weight_decay = arg.weight_decay
manual_name = arg.manual_name

aug_count = arg.aug_count

if use_topic_prob == True:
    with open("/home/zihan/fakenews/data_new/id_"+str(topic_num)+"TopicProb.pickle", "rb") as handle:
        print ("loading id_"+str(topic_num)+"TopicProb.pickle ...")
        id_topicProb = pickle.load(handle)
if use_publisher == True:
    with open("/home/zihan/fakenews/data_new/tr_pub_label.pickle", "rb") as handle:
        print ("loading tr_pub_label.pickle ...")
        tr_pub_label = pickle.load(handle)

if use_topic_feat == True:
    with open("/home/zihan/fakenews/data_new/id_"+str(topic_num)+"TopicFeat_train.pickle", "rb") as handle:
        id_topicFeat_train = pickle.load(handle)
    with open("/home/zihan/fakenews/data_new/id_"+str(topic_num)+"TopicFeat_val.pickle", "rb") as handle:
        id_topicFeat_test = pickle.load(handle)
    with open("/home/zihan/fakenews/data_new/id_"+str(topic_num)+"TopicProb.pickle", "rb") as handle:
        print ("loading id_"+str(topic_num)+"TopicProb.pickle ...")
        id_topicProb = pickle.load(handle)

USE_CUDA = arg.cuda and torch.cuda.is_available()
UNK_idx = 0
PAD_idx = 1
INVALID_idx = 0

pretrain_emb = arg.pretrain_emb
save_path = arg.save_path
test = arg.test

### lstm
n_layers = arg.n_layers
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
multiple_gpu = arg.multiple_gpu

## transformer 
parser.add_argument("--hop", type=int, default=6)
parser.add_argument("--heads", type=int, default=4)
parser.add_argument("--depth", type=int, default=40)
parser.add_argument("--filter", type=int, default=50)

### random
debug_mode = arg.debug_mode
build_vocab_flag = arg.build_vocab_flag
