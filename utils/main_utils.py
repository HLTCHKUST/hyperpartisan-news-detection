from utils import constant
import torch.nn as nn
import torch
# from pytorch_pretrained_bert import BertTokenizer, BertModel
import numpy as np
import time
import os
import pickle
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

def save_model(model, model_name, isFineTune=False, k=None):
    f_name= model_name+"_finetune" if isFineTune else model_name
    f_name= f_name+str(k) if k!=None else f_name
    
    folder_name = constant.save_path+constant.manual_name+"/"
    model_save_path = os.path.join(folder_name, f_name)
    if not os.path.isdir(folder_name):
        os.makedirs(folder_name)
    torch.save(model.state_dict(), model_save_path)
    print("Model saved in:", model_save_path)

def load_model(model, model_name):
    model_save_path = os.path.join(constant.save_path+constant.manual_name+"/", model_name)
    state = torch.load(model_save_path)
    model.load_state_dict(state)
    return model

def padding_for_bert(article, tit):
    maxLen_article = max(map(len, article))
    maxLen_article = 128 if maxLen_article > 128 else maxLen_article
    maxLen_tit = max(map(len, tit))
    maxLen_tit = 50 if maxLen_tit > 50 else maxLen_tit
    for i in range(len(article)):
        length = len(article[i])
        if length < maxLen_article:
            article[i].extend([0] * (maxLen_article - length))
        else:
            article[i] = article[i][0: maxLen_article]
    for i in range(len(tit)):
        length = len(tit[i])
        if length < maxLen_tit:
            tit[i].extend([0] * (maxLen_tit - length))
        else:
            tit[i] = tit[i][0: maxLen_tit]
    segments_ids_article = torch.LongTensor(np.zeros([len(article), maxLen_article]))
    segments_ids_tit = torch.LongTensor(np.zeros([len(tit), maxLen_tit]))
    return torch.LongTensor(article), segments_ids_article, torch.LongTensor(tit), segments_ids_tit

def evaluate(model, loader):
    model.eval()
    pred = []
    gold = []
    for X, x_len, tit, y, id_ in loader:
        pred_prob = model(X, x_len)
        pred.append(pred_prob[1].detach().cpu().numpy()) # ASK originally it was pred_prob[0]
        gold.append(y.cpu().numpy())

    pred = np.concatenate(pred)
    gold = np.concatenate(gold)
    
    accuracy, microPrecision, microRecall, microF1 = getMetrics(pred,gold,verbose=True)
    return accuracy

def eval_adver(lstm_model, classi_model, loader):
    pred = []
    gold = []
    for X, x_len, tit, y, id_ in loader:
        hidden_layer = lstm_model.feature(X, x_len)
        pred_prob_classi = classi_model(hidden_layer)
        pred_prob_classi = pred_prob_classi.view(len(pred_prob_classi))
        pred.append(pred_prob_classi.detach().cpu().numpy())
        gold.append(y.cpu().numpy())
    
    pred = np.concatenate(pred, axis=0)
    gold = np.concatenate(gold, axis=0)
    accuracy, microPrecision, microRecall, microF1 = getMetrics(pred,gold, verbose=True)
    return accuracy

def eval_adver2(lstm_model, classi_model, loader):
    pred = []
    gold = []
    for X, x_len, tit, y, _, topic_feat, id_ in loader:
        hidden_layer = lstm_model.feature(X, x_len)
        feature = torch.cat((hidden_layer, topic_feat), 1)
        pred_prob_classi = classi_model(feature)
        pred_prob_classi = pred_prob_classi.view(len(pred_prob_classi))
        pred.append(pred_prob_classi.detach().cpu().numpy())
        gold.append(y.cpu().numpy())
    
    pred = np.concatenate(pred, axis=0)
    gold = np.concatenate(gold, axis=0)
    accuracy, microPrecision, microRecall, microF1 = getMetrics(pred,gold, verbose=True)
    return accuracy

def eval_tit_lstm(article_model, title_model, LR, loader, use_add_feature_flag=False, writer=None, global_steps=None, isTest=False):
    pred = []
    gold = []
    id=[]
    if use_add_feature_flag:
        for X, x_len, tit, tit_len, emo2vec_feat, y, id_ in loader:
            article_feat = article_model.feature(X, x_len)
            title_feat = title_model.feature(tit, tit_len)
            feature = torch.cat((article_feat, title_feat), dim=1)
            feature = torch.cat((feature, emo2vec_feat), dim=1)
            pred_prob = LR(feature)
            pred_prob = pred_prob.view(len(pred_prob))
            pred.append(pred_prob.detach().cpu().numpy())
            gold.append(y.cpu().numpy())
            id.append(id_)
    else:
        for X, x_len, tit, tit_len, y, id_ in loader:
            article_feat = article_model.feature(X, x_len)
            title_feat = title_model.feature(tit, tit_len)
            feature = torch.cat((article_feat, title_feat), dim=1)
            pred_prob = LR(feature)
            pred_prob = pred_prob.view(len(pred_prob))
            pred.append(pred_prob.detach().cpu().numpy())
            gold.append(y.cpu().numpy())
            id.append(id_)
    pred = np.concatenate(pred, axis=0)
    gold = np.concatenate(gold, axis=0)
    id = np.concatenate(id, axis=0)
    
    accuracy, microPrecision, microRecall, microF1 = getMetrics(pred, gold, verbose=True)
    if writer!=None:
        phase_name='test' if isTest else 'val'
        writer.add_scalars(phase_name, {'acc': accuracy,'prec': microPrecision,'f1': microF1}, global_steps)
    return accuracy, pred, id

def eval_utransformer(article_model, title_model, LR, loader, use_add_feature_flag=False, writer=None, global_steps=None, isTest=False):
    pred = []
    gold = []
    if use_add_feature_flag:
        for X, x_len, tit, tit_len, emo2vec_feat, y, id_ in loader:
            article_feat = article_model(X)
            title_feat = title_model(tit)
            feature = torch.cat((article_feat, title_feat), dim=1)
            feature = torch.cat((feature, emo2vec_feat), dim=1)
            pred_prob = LR(feature)
            pred_prob = pred_prob.view(len(pred_prob))
            pred.append(pred_prob.detach().cpu().numpy())
            gold.append(y.cpu().numpy())
    else:
        for X, x_len, tit, tit_len, y, id_ in loader:
            article_feat = article_model(X)
            title_feat = title_model(tit)
            feature = torch.cat((article_feat, title_feat), dim=1)
            pred_prob = LR(feature)
            pred_prob = pred_prob.view(len(pred_prob))
            pred.append(pred_prob.detach().cpu().numpy())
            gold.append(y.cpu().numpy())
    pred = np.concatenate(pred, axis=0)
    gold = np.concatenate(gold, axis=0)
    accuracy, microPrecision, microRecall, microF1 = getMetrics(pred, gold, verbose=True)
    if writer!=None:
        phase_name='test' if isTest else 'val'
        writer.add_scalars(phase_name, {'acc': accuracy,'prec': microPrecision,'f1': microF1}, global_steps)
    return accuracy

def eval_bert(article_model, title_model, LR, loader, tokenizer, lstm_article=None, lstm_title=None, use_add_feature_flag=False, writer=None, global_steps=None, isTest=False):
    pred = []
    gold = []
    id=[]
    if use_add_feature_flag:
        for X, x_len, tit, tit_len, emo2vec_feat, y, id_ in tqdm(loader):
            X = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(item)) for item in X]
            tit = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(item)) for item in tit]
            # padding
            X, segments_ids_article, tit, segments_ids_tit = padding_for_bert(X, tit)
            if constant.USE_CUDA:
                X, segments_ids_article, tit, segments_ids_tit, y = X.cuda(), segments_ids_article.cuda(), tit.cuda(), segments_ids_tit.cuda(), y.cuda()
            #encoded_layer:[batch_size, sequence_length, hidden_size]
            encoded_article_layers, _ = article_model(X, segments_ids_article)
            article_feat = torch.sum(encoded_article_layers[-1], dim=1)
            encoded_tit_layers, _ = title_model(tit, segments_ids_tit)
            title_feat = torch.sum(encoded_tit_layers[-1], dim=1) #[batch_size, hidden_size]
            feature = torch.cat((article_feat, title_feat), dim=1)
            feature = torch.cat((feature, emo2vec_feat), dim=1)
            pred_prob = LR(feature)
            pred_prob = pred_prob.view(len(pred_prob))
            pred.append(pred_prob.detach().cpu().numpy())
            gold.append(y.cpu().numpy())
            id.append(id_)
    else:
        for X, x_len, tit, tit_len, y, id_ in tqdm(loader):
            X = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(item)) for item in X]
            tit = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(item)) for item in tit]
            # padding
            X, segments_ids_article, tit, segments_ids_tit = padding_for_bert(X, tit)
            if constant.USE_CUDA:
                X, segments_ids_article, tit, segments_ids_tit, y = X.cuda(), segments_ids_article.cuda(), tit.cuda(), segments_ids_tit.cuda(), y.cuda()
            #encoded_layer:[batch_size, sequence_length, hidden_size]
            encoded_article_layers, _ = article_model(X, segments_ids_article)
            encoded_tit_layers, _ = title_model(tit, segments_ids_tit)
            if constant.train_cleaner_dataset or constant.use_bert_plus_lstm:
                _, article_hidden = lstm_article(encoded_article_layers[-1])
                _, title_hidden = lstm_title(encoded_tit_layers[-1])
                article_feat = article_hidden[-1][-1]
                title_feat = title_hidden[-1][-1]
            else:
                article_feat = torch.sum(encoded_article_layers[-1], dim=1)
                title_feat = torch.sum(encoded_tit_layers[-1], dim=1) #[batch_size, hidden_size]
            feature = torch.cat((article_feat, title_feat), dim=1)
            pred_prob = LR(feature)
            pred_prob = pred_prob.view(len(pred_prob))
            pred.append(pred_prob.detach().cpu().numpy())
            gold.append(y.cpu().numpy())
            id.append(id_)
    pred = np.concatenate(pred, axis=0)
    gold = np.concatenate(gold, axis=0)
    id = np.concatenate(id, axis=0)

    accuracy, microPrecision, microRecall, microF1 = getMetrics(pred, gold, verbose=True)

    label2is_hyper = ["false", "true"]
    pred = (pred > 0.5) * 1
    with open('{}/{}'.format("nfolds_prediction", 'predict_'+str(accuracy)+'.txt'), 'w') as result_file:
        for idx, pre in zip(id, pred):
            pred_info = "{}\t{}\n".format(idx, label2is_hyper[pre])
            result_file.write(pred_info)
        print("file predict_{}.txt has been written into nfolds_prediction folder".format(accuracy))
    if writer!=None:
        phase_name='test' if isTest else 'val'
        writer.add_scalars(phase_name, {'acc': accuracy,'prec': microPrecision,'f1': microF1}, global_steps)
    
    return accuracy, pred, id

def eval_bert_with_chunked_data(article_model, title_model, LR, loader, tokenizer, ids_dict, gru=None, writer=None, global_steps=None, isTest=False):
    chunked_pred, chunked_gold, chunked_id_list= [], [], []
    for X, x_len, tit, tit_len, y, id_ in loader:
        if constant.use_bert_plus_gru:
            X = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(item)) for item in X]
            tit = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(item)) for item in tit]
            X, segments_ids_article, tit, segments_ids_tit = padding_for_bert(X, tit)
            if constant.USE_CUDA:
                X, segments_ids_article, tit, segments_ids_tit, y = X.cuda(), segments_ids_article.cuda(), tit.cuda(), segments_ids_tit.cuda(), y.cuda()
            encoded_article_layers, _ = article_model(X, segments_ids_article)
            encoded_tit_layers, _ = title_model(tit, segments_ids_tit)
            _, article_hidden = gru(encoded_article_layers[-1])
            _, title_hidden = gru(encoded_tit_layers[-1])
            article_feat = article_hidden[-1]
            title_feat = title_hidden[-1]
        else:
            X = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(item)) for item in X]
            tit = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(item)) for item in tit]
            X, segments_ids_article, tit, segments_ids_tit = padding_for_bert(X, tit)
            if constant.USE_CUDA:
                X, segments_ids_article, tit, segments_ids_tit, y = X.cuda(), segments_ids_article.cuda(), tit.cuda(), segments_ids_tit.cuda(), y.cuda()
            #encoded_layer:[batch_size, sequence_length, hidden_size]
            encoded_article_layers, _ = article_model(X, segments_ids_article)
            article_feat = torch.sum(encoded_article_layers[-1], dim=1)
            encoded_tit_layers, _ = title_model(tit, segments_ids_tit)
            title_feat = torch.sum(encoded_tit_layers[-1], dim=1) #[batch_size, hidden_size]
        feature = torch.cat((article_feat, title_feat), dim=1)
        pred_prob = LR(feature)
        pred_prob = pred_prob.view(len(pred_prob))
        chunked_pred.append(pred_prob.detach().cpu().numpy())
        chunked_gold.append(y.cpu().numpy())
        chunked_id_list.append(id_)
    chunked_pred = np.concatenate(chunked_pred, axis=0)
    chunked_gold = np.concatenate(chunked_gold, axis=0)
    chunked_id_list = list(np.concatenate(chunked_id_list, axis=0))
    pred, gold, ids = [], [], []
    for id_, chunked_ids in ids_dict.items():
        ids.append(id_)
        cnt_true, cnt_false = 0, 0
        # if cnt_true == cnt_false then depends on sigmoid_value
        sigmoid_value = 0
        for i, idx in enumerate(chunked_ids):
            index = chunked_id_list.index(idx)
            if i == 0: gold.append(chunked_gold[index])
            sigmoid_value += chunked_pred[index]
            if chunked_pred[index] > 0.5:
                cnt_true += 1
            else:
                cnt_false += 1
        # print("true:", cnt_true, "false:", cnt_false, "sigmoid value:", sigmoid_value, "gold:", gold[-1])
        if cnt_true > cnt_false:
            pred.append(1)
        elif cnt_true < cnt_false:
            pred.append(0)
        else:
            # average sigmoid value should be larger than 0.5
            if sigmoid_value * 1.0 / (cnt_true + cnt_false) > 0.5:
                pred.append(1)
            else:
                pred.append(0)
    pred = np.array(pred)
    gold = np.array(gold)
    ids = np.array(ids)
    accuracy, microPrecision, microRecall, microF1 = getMetrics(pred, gold, verbose=True)
    if writer!=None:
        phase_name='test' if isTest else 'val'
        writer.add_scalars(phase_name, {'acc': accuracy,'prec': microPrecision,'f1': microF1}, global_steps)
    return accuracy, pred, ids

def eval_tit_hier(hier_model, title_model, LR, loader, writer=None, global_steps=None, isTest=False):
    pred = []
    gold = []
    for X, tit, tit_len, y, id_ in loader:
        hier_feat = hier_model(X)
        title_feat = title_model.feature(tit, tit_len)
        feature = torch.cat((hier_feat, title_feat), dim=1)
        pred_prob = LR(feature)
        pred_prob = pred_prob.view(len(pred_prob))
        pred.append(pred_prob.detach().cpu().numpy())
        gold.append(y.cpu().numpy())
    
    pred = np.concatenate(pred, axis=0)
    gold = np.concatenate(gold, axis=0)
    accuracy, microPrecision, microRecall, microF1 = getMetrics(pred, gold, verbose=True)
    if writer != None:
        phase_name = 'test' if isTest else 'val'
        writer.add_scalars(phase_name, {'acc': accuracy,'prec': microPrecision,'f1': microF1}, global_steps)
    return accuracy

# predict for basic LSTM (need to add more to fit into different kinds of models)
def predict(article_model, title_model, LR, loader, k=None, name="", print_pred=False):
    label2is_hyper = ["false","true"]

    f_name= "predict{}.txt".format("_"+name)
    if k!=None:
        f_name="predict_{}.txt".format(k)
    with open('test/inputRun/{}'.format(f_name), 'w') as result_file:
        if not constant.use_bert:
            # result_file.write("id\tlabel\n")
            for X, x_len, tit, tit_len, _, id_ in tqdm(loader):
                article_feat = article_model.feature(X, x_len)
                title_feat = title_model.feature(tit, tit_len)
                feature = torch.cat((article_feat, title_feat), dim=1)
                pred_prob = LR(feature)
                pred_prob = pred_prob.view(len(pred_prob))
                preds = pred_prob.detach().cpu().numpy()

                # print out
                if k==None:
                    preds = (preds > 0.5) * 1
                    if print_pred:
                        for idx, pred, prob in zip(id_, preds, pred_prob):
                            pred_info = "{}\t{}\t{}\n".format(idx, prob, label2is_hyper[pred])
                            print(pred_info)
                            result_file.write(pred_info)
                    else:
                        for idx, pred in zip(id_, preds):
                            pred_info = "{}\t{}\n".format(idx, label2is_hyper[pred])
                            result_file.write(pred_info)
                else:
                    for idx, pred in zip(id_, preds):
                        pred_info = "{}\t{}\n".format(idx, pred)
                        result_file.write(pred_info)
        else:
            for X, x_len, tit, tit_len, _, id_ in tqdm(loader):
                from pytorch_pretrained_bert import BertTokenizer
                tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
                X = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(item)) for item in X]
                tit = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(item)) for item in tit]
                # padding
                X, segments_ids_article, tit, segments_ids_tit = padding_for_bert(X, tit)
                if constant.USE_CUDA:
                    X, segments_ids_article, tit, segments_ids_tit = X.cuda(), segments_ids_article.cuda(), tit.cuda(), segments_ids_tit.cuda()
                #encoded_layer:[batch_size, sequence_length, hidden_size]
                encoded_article_layers, _ = article_model(X, segments_ids_article)
                article_feat = torch.sum(encoded_article_layers[-1], dim=1)
                encoded_tit_layers, _ = title_model(tit, segments_ids_tit)
                title_feat = torch.sum(encoded_tit_layers[-1], dim=1) #[batch_size, hidden_size]
                feature = torch.cat((article_feat, title_feat), dim=1)
                pred_prob = LR(feature)
                pred_prob = pred_prob.view(len(pred_prob))
                preds = pred_prob.detach().cpu().numpy()

                # print out
                if k==None:
                    preds = (preds > 0.5) * 1
                    if print_pred:
                        for idx, pred, prob in zip(id_, preds, pred_prob):
                            pred_info = "{}\t{}\t{}\n".format(idx, prob, label2is_hyper[pred])
                            result_file.write(pred_info)
                    else:
                        for idx, pred in zip(id_, preds):
                            pred_info = "{}\t{}\n".format(idx, label2is_hyper[pred])
                            result_file.write(pred_info)
                else:
                    for idx, pred in zip(id_, preds):
                        pred_info = "{}\t{}\n".format(idx, pred)
                        result_file.write(pred_info)
    print("FILE {} SAVED".format(f_name))

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]

def getMetrics(predictions, ground, verbose=False):
    """Given predicted labels and the respective ground truth labels, display some metrics
    Input: shape [# of samples, NUM_CLASSES]
        predictions : Model output. Every row has 4 decimal values, with the highest belonging to the predicted class
        ground : Ground truth labels, converted to one-hot encodings. A sample belonging to Happy class will be [0, 1, 0, 0]
    Output:
        accuracy : Average accuracy
        microPrecision : Precision calculated on a micro level. Ref - https://datascience.stackexchange.com/questions/15989/micro-average-vs-macro-average-performance-in-a-multiclass-classification-settin/16001
        microRecall : Recall calculated on a micro level
        microF1 : Harmonic mean of microPrecision and microRecall. Higher value implies better classification  
    """
    
#     print(ground)
#     print(predictions.argmax(axis=1))
    
#     label2is_hyper = {0:"false", 1:"true"}
#     # [0.1, 0.3] -> [0, 1]
#     discretePredictions = to_categorical(predictions.argmax(axis=1),num_classes=2)
    # predictions = predictions.argmax(axis=1)
    predictions = (predictions > 0.5) * 1
    # print (predictions)
    accuracy = accuracy_score(ground, predictions)
    precision = precision_score(ground, predictions)
    recall = recall_score(ground, predictions)
    f1 = f1_score(ground, predictions)
    
    if(verbose):
        print ("confusion matrix:")
        print (confusion_matrix(ground, predictions))
        print("accuracy : %.4f, precision : %.4f, recall : %.4f, f1_score : %.4f" % (accuracy, precision, recall, f1))
        
    return accuracy, precision, recall, f1
