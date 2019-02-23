import torch
import torch.nn as nn
from torch.utils import data
from torch.optim.lr_scheduler import StepLR
import models
from utils import constant
from utils.new_xml_data_parsing import read_xml, parse_xml

from utils.data_reader_online import prepare_data_tira
from utils.main_utils import save_model, load_model, predict, getMetrics, eval_tit_lstm, eval_bert, padding_for_bert, eval_bert_with_chunked_data, padding_for_bert

import pickle
from tqdm import tqdm
import numpy as np

def read_input_xml(f_path):
    # prprocessed data
#     data_articles, gold_articles = read_xml(f_path) # on local only. i.e. when we have gold label

    data_articles = read_xml(f_path)
    preprocessed_dict={}
    label_dic={}
    ids = []
    for item in data_articles:
        parsed = parse_xml(item)       
        id_ = parsed[0]
        
        ids.append(id_)
        preprocessed_dict[id_] = parsed[1]
        
#     for item in gold_articles:
#         id_ = item.get('id')
#         label = item.get('hyperpartisan')
#         label_dic[id_] = label
        
    with open('/home/yeon-zi/fakenews_submission/data/preprocessed.pickle', 'wb') as handle:
        pickle.dump(preprocessed_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('/home/yeon-zi/fakenews_submission/data/ids.pickle', 'wb') as handle:
        pickle.dump(ids, handle, protocol=pickle.HIGHEST_PROTOCOL)
#     with open('./data/labels.pickle', 'wb') as handle:
#         pickle.dump(label_dic, handle, protocol=pickle.HIGHEST_PROTOCOL) 

def test():
    # prepare data_loader and vocab
    print("prepare start")
    data_loader_test, vocab = prepare_data_tira()
    print("prepare end")
    if constant.use_bert:
        if constant.bert_from_scratch:
            # bert_model = torch.load("/home/yeon-zi/fakenews_submission/bert_model/whole_bert_without_finetune.bin")
            
            # use fine tune bert
            bert_model = torch.load("/home/yeon-zi/fakenews_submission/bert_model/whole_bert_model.model")
            LR = models.Classifier(hidden_dim1=768, hidden_dim2=768)
        else:
            bert_model = torch.load("/home/yeon-zi/fakenews_submission/bert_model/whole_bert_model.model")
            lstm_article = nn.LSTM(input_size=768, hidden_size=constant.hidden_dim, 
                                num_layers=constant.n_layers, bidirectional=False, batch_first=True)
            lstm_title = nn.LSTM(input_size=768, hidden_size=constant.hidden_dim_tit,
                                num_layers=constant.n_layers, bidirectional=False, batch_first=True)
            LR = models.Classifier(hidden_dim1=constant.hidden_dim, hidden_dim2=constant.hidden_dim_tit)
        if constant.USE_CUDA:
            classifer_state = torch.load("/home/yeon-zi/fakenews_submission/bert_model/fold_1_classifier.bin")
            lstm_article_state = torch.load("/home/yeon-zi/fakenews_submission/bert_model/fold_1_lstm_article.bin")
            lstm_title_state = torch.load("/home/yeon-zi/fakenews_submission/bert_model/fold_1_lstm_title.bin")
        else:
            if constant.bert_from_scratch:
                # classifer_state = torch.load("/home/yeon-zi/fakenews_submission/bert_model/finetune_classi_0.72.bin", map_location=lambda storage, loc: storage)
                
                # pair with fine tune bert
                classifer_state = torch.load("/home/yeon-zi/fakenews_submission/bert_model/finetune_classi_for_tunebert_0.76.bin", map_location=lambda storage, loc: storage)
            else:
                classifer_state = torch.load("/home/yeon-zi/fakenews_submission/bert_model/fold_1_classifier.bin", map_location=lambda storage, loc: storage)
                lstm_article_state = torch.load("/home/yeon-zi/fakenews_submission/bert_model/fold_1_lstm_article.bin", map_location=lambda storage, loc: storage)
                lstm_title_state = torch.load("/home/yeon-zi/fakenews_submission/bert_model/fold_1_lstm_title.bin", map_location=lambda storage, loc: storage)

        article_model = bert_model
        title_model = bert_model
        if constant.bert_from_scratch:
            LR.load_state_dict(classifer_state)
        else:
            lstm_article.load_state_dict(lstm_article_state)
            lstm_title.load_state_dict(lstm_title_state)
            LR.load_state_dict(classifer_state)
    else:
        # for basic LSTM model
        article_model = models.LSTM(vocab=vocab, 
                        embedding_size=constant.emb_dim, 
                        hidden_size=constant.hidden_dim, 
                        num_layers=constant.n_layers,
                        pretrain_emb=constant.pretrain_emb
                        )
        title_model = models.LSTM(vocab=vocab,
                        embedding_size=constant.emb_dim,
                        hidden_size=constant.hidden_dim_tit,
                        num_layers=constant.n_layers,
                        pretrain_emb=constant.pretrain_emb
                        )
        LR = models.Classifier(hidden_dim1=constant.hidden_dim, hidden_dim2=constant.hidden_dim_tit)

        # load parameters
       # article_model = load_model(article_model, model_name="article_model", map_location=lambda storage, loc: storage)
       # title_model = load_model(title_model, model_name="title_model", map_location=lambda storage, loc: storage)
       # LR = load_model(LR, model_name="LR", map_location=lambda storage, loc: storage)

        lr_state = torch.load("/home/yeon-zi/fakenews_submission/pub_model/finetune_classi_for_lstm_0.67.bin", map_location=lambda storage, loc: storage)
        article_state = torch.load("/home/yeon-zi/fakenews_submission/pub_model/article_model", map_location=lambda storage, loc: storage)
        title_state = torch.load("/home/yeon-zi/fakenews_submission/pub_model/title_model", map_location=lambda storage, loc: storage)

        article_model.load_state_dict(article_state)
        title_model.load_state_dict(title_state)
        LR.load_state_dict(lr_state)

    if constant.USE_CUDA:
        article_model.cuda()
        title_model.cuda()
        if not constant.bert_from_scratch:
            lstm_article.cuda()
            lstm_title.cuda()
        LR.cuda()
    
    # predict and save result in result folder
    if constant.use_bert:
        article_model.eval()
        title_model.eval()
        if not constant.bert_from_scratch:
            lstm_article.eval()
            lstm_title.eval()
        LR.eval()
        if constant.bert_from_scratch:
            predict(article_model, title_model, None, None, LR, data_loader_test, output_path=constant.output_path)
        else:
            predict(article_model, title_model, lstm_article, lstm_title, LR, data_loader_test, output_path=constant.output_path)
    else:
        article_model.eval()
        title_model.eval()
        LR.eval()
        predict(article_model, title_model, None, None, LR, data_loader_test, output_path=constant.output_path)
if __name__ == "__main__":
    read_input_xml(constant.input_path)
    test()
