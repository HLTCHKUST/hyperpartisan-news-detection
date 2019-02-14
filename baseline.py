import models
from models.discriminator import LR
from pytorch_pretrained_bert import BertTokenizer, BertModel
from utils import constant
from utils.data_reader_online import prepare_data
from utils.main_utils import predict, evaluate, getMetrics, save_model, load_model, eval_tit_lstm, eval_bert, eval_utransformer, padding_for_bert
from utils.feature_utils import create_embedding_matrix, gen_embeddings
import torch.nn as nn
import torch
import numpy as np
from tqdm import tqdm

from tensorboardX import SummaryWriter

def train(article_model, title_model, LR, data_loader_train, data_loader_val, data_loader_test, vocab, tokenizer=None):
    """ 
    Training loop
    Inputs:
        model: the model to be trained
        data_loader_train: training data loader
        data_loader_val: validation data loader
        vocab: vocabulary list
    Output:
        avg_best: best f1 score on validation data
    """
    if(constant.USE_CUDA): 
        article_model.cuda()
        title_model.cuda()
        LR.cuda()
        print("using cuda")
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCELoss()
    
    if constant.use_bert:
        experiment_name = '{0}_LR_lr{1}_weight_decay_{2}'.format(constant.manual_name, constant.lr_classi, constant.weight_decay)
    elif constant.use_utransformer:
        experiment_name = '{0}_ArLr{1}_TitLr{2}_LRLr{3}_ArHop{4}_TitHop{5}_ArSize{6}_TitSize{7}_input_drop{8}_layer_drop{9}_atten_drop{10}_relu_drop{11}_weight_decay{12}'.format(constant.manual_name, constant.lr_lstm, constant.lr_title, constant.lr_classi, constant.max_hops_article, constant.max_hops_title, constant.filter_size_article, constant.filter_size_title, constant.input_dropout, constant.layer_dropout, constant.attention_dropout, constant.relu_dropout,constant.weight_decay)
    else:
        experiment_name = '{0}_lstm_alr{1}_tlr{2}_lrlr{3}'.format(constant.manual_name, constant.lr_lstm, constant.lr_title, constant.lr_classi)
    logdir = "tensorboard/" + experiment_name + "/"
    writer = SummaryWriter(logdir)
    
    if constant.use_bert:
        model = [
            {"params": article_model.parameters(), "lr": constant.lr_lstm},
            # {"params": title_model.parameters(), "lr": constant.lr_title},
            {"params": LR.parameters(), "lr": constant.lr_classi}
        ]
    else:
        model = [
            {"params": article_model.parameters(), "lr": constant.lr_lstm},
            {"params": title_model.parameters(), "lr": constant.lr_title},
            {"params": LR.parameters(), "lr": constant.lr_classi}
        ]
    if constant.optimizer=='adam':
        opt = torch.optim.Adam(model, lr=constant.lr, weight_decay=constant.weight_decay)
    elif constant.optimizer=='adagrad':
        opt = torch.optim.Adagrad(model, lr=constant.lr)
    elif constant.optimizer=='sgd':
        opt = torch.optim.SGD(model, lr=constant.lr, momentum=0.9)

    avg_best = 0
    test_best = 0
    cnt = 0
    global_steps = 0
    for e in range(constant.max_epochs):
        article_model.train()
        title_model.train()
        loss_log = []
        f1_log = 0
        acc_log = 0

        pbar = tqdm(enumerate(data_loader_train),total=len(data_loader_train))

        for i, (X, x_len, tit, tit_len, y, ind) in pbar:
            opt.zero_grad()
            if constant.use_bert:
                tit = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(item)) for item in tit]
                # padding
                tit, segments_ids_tit = padding_for_bert(tit)
                if constant.USE_CUDA:
                    X, tit, segments_ids_tit, y = X.cuda(), tit.cuda(), segments_ids_tit.cuda(), y.cuda()
                #encoded_layer:[batch_size, sequence_length, hidden_size]
                article_feat = article_model.feature(X, x_len)
                encoded_tit_layers, _ = title_model(tit, segments_ids_tit)
                title_feat = torch.sum(encoded_tit_layers[-1], dim=1) #[batch_size, hidden_size]
            elif constant.use_utransformer:
                article_feat = article_model(X)
                title_feat = title_model(tit)
            else:
                article_feat = article_model.feature(X, x_len)
                title_feat = title_model.feature(tit, tit_len)
            feature = torch.cat((article_feat, title_feat), dim=1)
            pred_prob = LR(feature)

            loss = criterion(pred_prob, y)
            loss.backward()
            opt.step()
            ## logging 
            loss_log.append(loss.item())
            accuracy, microPrecision, microRecall, microF1 = getMetrics(pred_prob.detach().cpu().numpy(), y.cpu().numpy())
            f1_log += microF1
            acc_log += accuracy
            pbar.set_description("(Epoch {}) TRAIN F1:{:.4f} TRAIN LOSS:{:.4f} ACCURACY:{:.4f}".format((e+1), f1_log/float(i+1), np.mean(loss_log), acc_log/float(i+1)))
            
            writer.add_scalars('train', {'loss': np.mean(loss_log),'acc': acc_log/float(i+1),
                                            'f1': f1_log/float(i+1)}, global_steps)
            global_steps+=1
            
        ## LOG
        if(e % 1 == 0):
            print("Evaluation on Val")
            use_add_feature_flag = constant.use_emo2vec_feat or constant.use_url
            if constant.use_bert:
                accuracy = eval_bert(article_model, title_model, LR, data_loader_val, tokenizer, use_add_feature_flag, writer, e)
            elif constant.use_utransformer:
                accuracy = eval_utransformer(article_model, title_model, LR, data_loader_val, use_add_feature_flag, writer, e)
            else:
                accuracy = eval_tit_lstm(article_model, title_model, LR, data_loader_val, use_add_feature_flag, writer, e)
            if(accuracy > avg_best):
                avg_best = accuracy
                
                print("Evaluation on Testset")
                if constant.use_bert:
                    test_acc = eval_bert(article_model, title_model, LR, data_loader_test, tokenizer, use_add_feature_flag, writer, e, True)
                elif constant.use_utransformer:
                    test_acc = eval_utransformer(article_model, title_model, LR, data_loader_test, use_add_feature_flag, writer, e, True)
                else:
                    test_acc = eval_tit_lstm(article_model, title_model, LR, data_loader_test, use_add_feature_flag, writer, e, True)
                if test_acc > test_best:
                    test_best = test_acc
                    print("Find better model. Saving model ...")
                    save_model(article_model, "article_model")
                    save_model(title_model, "title_model")
                    save_model(LR, "LR")
#                 predict(model, criterion, data_loader_dev_no_lab) ## print the prediction with the highest Micro-F1
                cnt = 0
            else:
                cnt += 1
            if(cnt == 3): break
            if(avg_best == 1.0): break

    return avg_best, test_best

data_loader_tr, data_loader_val, data_loader_test, vocab = prepare_data('/home/nayeon/fakenews/data_new/preprocessed_new_{}_wtitle.pickle', constant.batch_size)

if constant.use_bert:
    article_model = models.LSTM(vocab=vocab, 
                    embedding_size=constant.emb_dim, 
                    hidden_size=constant.hidden_dim, 
                    num_layers=constant.n_layers,
                    pretrain_emb=constant.pretrain_emb
                    )
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    title_model = BertModel.from_pretrained('bert-base-uncased')
    LR = models.LR(hidden_dim1=constant.hidden_dim, hidden_dim2=768)
elif constant.use_utransformer:
    article_model = models.UTransformer(vocab=vocab,
                    embedding_size=constant.emb_dim,
                    hidden_size=constant.hidden_dim,
                    num_layers=constant.max_hops_article,
                    num_heads=constant.num_heads,
                    total_key_depth=constant.key_value_depth,
                    total_value_depth=constant.key_value_depth,
                    filter_size=constant.filter_size_article,
                    input_dropout=constant.input_dropout,
                    layer_dropout=constant.layer_dropout,
                    attention_dropout=constant.attention_dropout,
                    relu_dropout=constant.relu_dropout
                    )
    title_model = models.UTransformer(vocab=vocab,
                    embedding_size=constant.emb_dim,
                    hidden_size=constant.hidden_dim_tit,
                    num_layers=constant.max_hops_title,
                    num_heads=constant.num_heads,
                    total_key_depth=constant.key_value_depth,
                    total_value_depth=constant.key_value_depth,
                    filter_size=constant.filter_size_title,
                    input_dropout=constant.input_dropout,
                    layer_dropout=constant.layer_dropout,
                    attention_dropout=constant.attention_dropout,
                    relu_dropout=constant.relu_dropout
                    )
    LR = models.LR(hidden_dim1=constant.hidden_dim, hidden_dim2=constant.hidden_dim_tit)
else:
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
    LR = models.LR(hidden_dim1=constant.hidden_dim, hidden_dim2=constant.hidden_dim_tit)

if constant.use_bert:
    avg_best, test_best = train(article_model, title_model, LR, data_loader_tr, data_loader_val, data_loader_test, vocab, tokenizer=tokenizer)
else:
    avg_best, test_best = train(article_model, title_model, LR, data_loader_tr, data_loader_val, data_loader_test, vocab)
print("Best VAL ACC: %3.5f, Best TEST ACC: %3.5f" % (avg_best, test_best))