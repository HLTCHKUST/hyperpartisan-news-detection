import torch
import torch.nn as nn
from torch.utils import data
from torch.optim.lr_scheduler import StepLR
import models
from utils import constant
from utils.main_utils import save_model, load_model, predict, getMetrics, eval_tit_lstm, eval_bert, eval_bert_with_chunked_data, padding_for_bert
from utils.data_reader_online import prepare_data, prepare_byarticle_data, prepare_byarticle_cross_validation, prepare_filtered_data, prepare_nfolds_cleaner_data

import pickle
from tqdm import tqdm
import numpy as np

from sklearn.model_selection import KFold

from tensorboardX import SummaryWriter

def split_data():
    """
        test set: 50 true, 50 false
        train set: 188 true, 357 false
    """
    print("spliting data ...")
    ids_file = open("data_new/by_article_ids.pickle", "rb")
    ids = pickle.load(ids_file)
    data_file = open("data_new/preprocessed_byarticle_data.pickle", "rb")
    preprocessed_data = pickle.load(data_file)
    labels_file = open("data_new/by_article_labels.pickle", "rb")
    labels = pickle.load(labels_file)
    preprocessed_byarticle_train, preprocessed_byarticle_test = {}, {}
    ids_train, ids_test = [], []
    labels_train, labels_test = {}, {}
    cnt_false, cnt_true = 0, 0
    for id_ in ids:
        if labels[id_] == "true":
            if cnt_true < 50:
                cnt_true += 1
                ids_test.append(id_)
                labels_test[id_] = "true"
                preprocessed_byarticle_test[id_] = preprocessed_data[id_]
            else:
                ids_train.append(id_)
                labels_train[id_] = "true"
                preprocessed_byarticle_train[id_] = preprocessed_data[id_]
        else:
            if cnt_false < 50:
                cnt_false += 1
                ids_test.append(id_)
                labels_test[id_] = "false"
                preprocessed_byarticle_test[id_] = preprocessed_data[id_]
            else:
                ids_train.append(id_)
                labels_train[id_] = "false"
                preprocessed_byarticle_train[id_] = preprocessed_data[id_]
    print("dumping ...")
    pkl_out = open("data_new/by_article_ids_train.pickle", "wb")
    pickle.dump(ids_train, pkl_out)
    pkl_out = open("data_new/by_article_ids_test.pickle", "wb")
    pickle.dump(ids_test, pkl_out)
    pkl_out = open("data_new/by_article_labels_train.pickle", "wb")
    pickle.dump(labels_train, pkl_out)
    pkl_out = open("data_new/by_article_labels_test.pickle", "wb")
    pickle.dump(labels_test, pkl_out)
    pkl_out = open("data_new/preprocessed_byarticle_train.pickle", "wb")
    pickle.dump(preprocessed_byarticle_train, pkl_out)
    pkl_out = open("data_new/preprocessed_byarticle_test.pickle", "wb")
    pickle.dump(preprocessed_byarticle_test, pkl_out)

def predict():
    # prepare data_loader and vocab
    use_by_article = False
    if use_by_article:
        _, data_loader_test, vocab = prepare_byarticle_data()
    else:
        _, _, data_loader_test, vocab = prepare_data('./data_new/preprocessed_new_{}', constant.batch_size)
    
    if constant.use_bert:
        from pytorch_pretrained_bert import BertTokenizer, BertModel
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        bert_model = BertModel.from_pretrained('bert-base-uncased')
        state = torch.load("bert_model/pytorch_model.bin")
        bert_model.load_state_dict(state)
        article_model = bert_model
        title_model = bert_model
        # print("finish bert model loading")
        LR = models.Classifier(hidden_dim1=768, hidden_dim2=768)
        classifer_state = torch.load("bert_model/classifier.bin")
        LR.load_state_dict(classifer_state)
        # 
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
        LR = models.LR(hidden_dim1=constant.hidden_dim, hidden_dim2=constant.hidden_dim_tit)

        # load parameters
        article_model = load_model(article_model, model_name="article_model")
        title_model = load_model(title_model, model_name="title_model")
        LR = load_model(LR, model_name="LR")

    if constant.USE_CUDA:
        article_model.cuda()
        title_model.cuda()
        LR.cuda()

    # predict and save result in result folder
    predict(article_model, title_model, LR, data_loader_test, name="bypublisher", print_pred=True)

def train(aug_count=""):
    # prepare data_loader and vocab
    if constant.train_cleaner_dataset:
        data_loader_train, data_loader_test, vocab = prepare_filtered_data(batch_size=constant.batch_size)
    else:
        data_loader_train, data_loader_test, vocab = prepare_byarticle_data(aug_count=aug_count, batch_size=constant.batch_size)
    
    # load parameters, LR is for fine tune
    if constant.use_bert:
        from pytorch_pretrained_bert import BertTokenizer, BertModel
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        bert_model = BertModel.from_pretrained('bert-base-uncased')
        if not constant.bert_from_scratch:
            state = torch.load("bert_model/pytorch_model.bin")
            bert_model.load_state_dict(state)
        article_model = bert_model
        title_model = bert_model
        # print("finish bert model loading")
        if constant.train_cleaner_dataset:
            lstm_article = nn.LSTM(input_size=768, hidden_size=constant.hidden_dim, 
                                   num_layers=constant.n_layers, bidirectional=False, batch_first=True)
            lstm_title = nn.LSTM(input_size=768, hidden_size=constant.hidden_dim_tit,
                                 num_layers=constant.n_layers, bidirectional=False, batch_first=True)
            LR = models.Classifier(hidden_dim1=constant.hidden_dim, hidden_dim2=constant.hidden_dim_tit)
        else:
            LR = models.Classifier(hidden_dim1=768, hidden_dim2=768)
    else:
        # for basic LSTM model
        article_model = models.LSTM(vocab=vocab, 
                        embedding_size=constant.emb_dim, 
                        hidden_size=constant.hidden_dim, 
                        num_layers=constant.n_layers,
                        pretrain_emb=constant.pretrain_emb,
                        use_attn=constant.use_attn
                        )
        title_model = models.LSTM(vocab=vocab,
                        embedding_size=constant.emb_dim,
                        hidden_size=constant.hidden_dim_tit,
                        num_layers=constant.n_layers,
                        pretrain_emb=constant.pretrain_emb,
                        use_attn=constant.use_attn
                        )
#         LR = models.LR(hidden_dim1=constant.hidden_dim, hidden_dim2=constant.hidden_dim_tit)
        LR = models.Classifier(hidden_dim1=constant.hidden_dim, hidden_dim2=constant.hidden_dim_tit)

        article_model = load_model(article_model, model_name="LstmAr_LstmTit_att_batchnorm_lstm_alr0.0005_tlr0.005_lrlr0.001_opadam_article_model")
        title_model = load_model(title_model, model_name="LstmAr_LstmTit_att_batchnorm_lstm_alr0.0005_tlr0.005_lrlr0.001_opadam_title_model")
        LR = load_model(LR, model_name="LstmAr_LstmTit_att_batchnorm_lstm_alr0.0005_tlr0.005_lrlr0.001_opadam_LR")
        
    if constant.USE_CUDA:
        article_model.cuda()
        title_model.cuda()
        LR.cuda()
        if constant.train_cleaner_dataset:
            lstm_article.cuda()
            lstm_title.cuda()

    criterion = nn.BCELoss()
    
    if constant.train_cleaner_dataset:
        model = [
                {"params": lstm_article.parameters(), "lr": constant.lr_lstm},
                {"params": lstm_title.parameters(), "lr": constant.lr_title}, 
                {"params": LR.parameters(), "lr": constant.lr_classi},
            ]
        if constant.optimizer=='adam':
            opt = torch.optim.Adam(model, lr=constant.lr_classi, weight_decay=constant.weight_decay)
        elif constant.optimizer=='adagrad':
            opt = torch.optim.Adagrad(model, lr=constant.lr_classi)
        elif constant.optimizer=='sgd':
            opt = torch.optim.SGD(model, lr=constant.lr_classi, momentum=0.9)
    else:
        if constant.optimizer=='adam':
            opt = torch.optim.Adam(LR.parameters(), lr=constant.lr_classi, weight_decay=constant.weight_decay)
        elif constant.optimizer=='adagrad':
            opt = torch.optim.Adagrad(LR.parameters(), lr=constant.lr_classi)
        elif constant.optimizer=='sgd':
            opt = torch.optim.SGD(LR.parameters(), lr=constant.lr_classi, momentum=0.9)

    # test the result without fine tune
    # print("testing without fine tune")
    # accuracy = eval_tit_lstm(article_model, title_model, LR, data_loader_test, False)

    # set tensorboard folder name
    if constant.use_bert:
        experiment_name = "BERT_FineTune_aug{0}_LRlr{1}".format(constant.aug_count, constant.lr_classi)
    else:
        experiment_name = "LSTM_FineTune_aug{0}_LRlr{1}".format(constant.aug_count, constant.lr_classi)
    
    logdir = "tensorboard/" + experiment_name + "/"
    writer = SummaryWriter(logdir)

    test_best = 0
    cnt = 0
    global_steps = 0
    for e in range(constant.max_epochs):
        article_model.train()
        title_model.train()
        LR.train()
        if constant.train_cleaner_dataset:
            lstm_article.train()
            lstm_title.train()
        loss_log = []
        f1_log = 0
        acc_log = 0

        # training
        pbar = tqdm(enumerate(data_loader_train),total=len(data_loader_train))
        for i, (X, x_len, tit, tit_len, y, ind) in pbar:
            opt.zero_grad()
            if constant.use_bert:
                X = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(item)) for item in X]
                tit = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(item)) for item in tit]
                # padding
                X, segments_ids_article, tit, segments_ids_tit = padding_for_bert(X, tit)
                if constant.USE_CUDA:
                    X, segments_ids_article, tit, segments_ids_tit, y = X.cuda(), segments_ids_article.cuda(), tit.cuda(), segments_ids_tit.cuda(), y.cuda()
                encoded_article_layers, _ = article_model(X, segments_ids_article)
                encoded_tit_layers, _ = title_model(tit, segments_ids_tit)
                if constant.train_cleaner_dataset:
                    _, article_hidden = lstm_article(encoded_article_layers[-1])
                    _, title_hidden = lstm_title(encoded_tit_layers[-1])
                    article_feat = article_hidden[-1][-1]
                    title_feat = title_hidden[-1][-1]
                else:
                    article_feat = torch.sum(encoded_article_layers[-1], dim=1)
                    title_feat = torch.sum(encoded_tit_layers[-1], dim=1) #[batch_size, hidden_size]
            else:
                article_feat = article_model.feature(X, x_len)
                title_feat = title_model.feature(tit, tit_len)
            feature = torch.cat((article_feat, title_feat), dim=1)
            pred_prob = LR(feature)
            
            loss = criterion(pred_prob, y)
            loss.backward()
            opt.step()

            loss_log.append(loss.item())
            accuracy, microPrecision, microRecall, microF1 = getMetrics(pred_prob.detach().cpu().numpy(), y.cpu().numpy())
            f1_log += microF1
            acc_log += accuracy
            pbar.set_description("(Epoch {}) TRAIN F1:{:.4f} TRAIN LOSS:{:.4f} ACCURACY:{:.4f}".format((e+1), f1_log/float(i+1), np.mean(loss_log), acc_log/float(i+1)))

            writer.add_scalars('train', {'loss': np.mean(loss_log),
                                         'acc': acc_log/float(i+1),
                                         'f1': f1_log/float(i+1)}, global_steps)
            global_steps+=1
        
        article_model.eval()
        title_model.eval()
        LR.eval()
        if constant.train_cleaner_dataset:
            lstm_article.eval()
            lstm_title.eval()
        # testing
        if(e % 1 == 0):
            print("Evaluation on Test")
            use_add_feature_flag = constant.use_emo2vec_feat or constant.use_url
            if constant.use_bert:
                if constant.train_cleaner_dataset:
                    accuracy, pred, id_ = eval_bert(article_model, title_model, LR, data_loader_test, tokenizer, lstm_article, lstm_title, use_add_feature_flag, writer, e, True)
                else:
                    accuracy, pred, id_ = eval_bert(article_model, title_model, LR, data_loader_test, tokenizer, None, None, use_add_feature_flag, writer, e, True)
            else:
                accuracy, pred, id_ = eval_tit_lstm(article_model, title_model, LR, data_loader_test, use_add_feature_flag, writer, e, True)
            
            if(accuracy > test_best):
                test_best = accuracy
                print("Find better model. Saving model ...")
                cnt = 0
                if constant.train_cleaner_dataset:
                    torch.save(lstm_article.state_dict(), "bert_model/by_publisher/lstm_article_"+str(constant.hidden_dim)+"_"+str(constant.hidden_dim_tit)+"_"+str(test_best)+".bin")
                    torch.save(lstm_title.state_dict(), "bert_model/by_publisher/lstm_title_"+str(constant.hidden_dim)+"_"+str(constant.hidden_dim_tit)+"_"+str(test_best)+".bin")
                    torch.save(LR.state_dict(), "bert_model/by_publisher/classifier_bypublisher_"+str(constant.hidden_dim)+"_"+str(constant.hidden_dim_tit)+"_"+str(test_best)+".bin")
                    print("The lstm_article lstm_title classifier_bypublisher have been saved!")
                else:
                    torch.save(LR.state_dict(), "bert_model/finetune_classi_"+str(accuracy)+".bin")
                    print("The fine tune classifier has been saved!")
            else:
                cnt += 1
            if(cnt == 10): 
                # save prediction and gold
                with open('pred/{0}_pred.pickle'.format(experiment_name), 'wb') as handle:
                    pickle.dump({"preds":pred, "ids":id_}, handle, protocol=pickle.HIGHEST_PROTOCOL)
                break
            if(test_best == 1.0): 
                # save prediction and gold
                with open('pred/{0}_pred.pickle'.format(experiment_name), 'wb') as handle:
                    pickle.dump({"preds":pred, "ids":id_}, handle, protocol=pickle.HIGHEST_PROTOCOL)
                break

def eval_bert_plus_lstm():
    data_loader_test = prepare_byarticle_data(aug_count=constant.aug_count, batch_size=constant.batch_size, test_whole=True)
    # load model
    # bert model
    from pytorch_pretrained_bert import BertTokenizer, BertModel
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    state = torch.load("bert_model/pytorch_model.bin")
    bert_model.load_state_dict(state)
    article_model = bert_model
    title_model = bert_model
    # lstm model and classifier
    lstm_article = nn.LSTM(input_size=768, hidden_size=constant.hidden_dim, 
                           num_layers=constant.n_layers, bidirectional=False, batch_first=True)
    lstm_title = nn.LSTM(input_size=768, hidden_size=constant.hidden_dim_tit,
                         num_layers=constant.n_layers, bidirectional=False, batch_first=True)
    classifier = models.Classifier(hidden_dim1=constant.hidden_dim, hidden_dim2=constant.hidden_dim_tit)
    lstm_article.load_state_dict(torch.load("bert_model/9folds_large/fold_3_lstm_article_0.9709711056544115.bin"))
    lstm_title.load_state_dict(torch.load("bert_model/9folds_large/fold_3_lstm_title_0.9709711056544115.bin"))
    classifier.load_state_dict(torch.load("bert_model/9folds_large/fold_3_classifier_0.9709711056544115.bin"))

    if constant.USE_CUDA:
        article_model.cuda()
        title_model.cuda()
        lstm_article.cuda()
        lstm_title.cuda()
        classifier.cuda()

    article_model.eval()
    title_model.eval()
    lstm_article.eval()
    lstm_title.eval()
    classifier.eval()
    accuracy, pred, id_ = eval_bert(article_model, title_model, classifier, data_loader_test, tokenizer, lstm_article, lstm_title, False, None, 1, True)

def cross_validation(kfold=10):
    with open("data_new/by_article_ids.pickle", "rb") as ids_file:
        ids = pickle.load(ids_file)
    with open("data_new/preprocessed_byarticle_data.pickle", "rb")as data_file:
        data = pickle.load(data_file)
    with open("data_new/by_article_labels.pickle", "rb") as labels_file:
        labels = pickle.load(labels_file)
    with open("/home/nayeon/fakenews/data_new/vocab_trim4.pickle", 'rb') as vocab_file:
        vocab = pickle.load(vocab_file)

    if not constant.use_bert:
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
        article_model = load_model(article_model, model_name="article_model")
        title_model = load_model(title_model, model_name="title_model")
    else:
        from pytorch_pretrained_bert import BertTokenizer, BertModel
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        bert_model = BertModel.from_pretrained('bert-base-uncased')
        if not constant.bert_from_scratch:
            state = torch.load("bert_model/pytorch_model.bin")
            bert_model.load_state_dict(state)
        article_model = bert_model
        title_model = bert_model
        if constant.use_bert_plus_lstm:
            lstm_article = nn.LSTM(input_size=768, hidden_size=constant.hidden_dim, 
                                   num_layers=constant.n_layers, bidirectional=False, batch_first=True)
            lstm_title = nn.LSTM(input_size=768, hidden_size=constant.hidden_dim_tit,
                                 num_layers=constant.n_layers, bidirectional=False, batch_first=True)
            lstm_article.load_state_dict(torch.load("bert_model/lstm_article2.bin"))
            lstm_title.load_state_dict(torch.load("bert_model/lstm_title2.bin"))
    
    # set average test acc
    avg_test_acc = 0
    best_acc = 0
    k = 0
    kf = KFold(n_splits=kfold)
    for train_index, test_index in kf.split(ids):
        k += 1
        print("k:", k)
        # get 25 true 25 false for validation #
        ids_train, ids_val = [], []
        data_train, data_val = {}, {}
        labels_train, labels_val = {}, {}
        cnt_true, cnt_false = 0, 0
        for index in train_index:
            id_ = ids[index]
            if labels[id_] == "true":
                if cnt_true < 25:
                    cnt_true += 1
                    ids_val.append(id_)
                    data_val[id_] = data[id_]
                    labels_val[id_] = labels[id_]
                else:
                    ids_train.append(id_)
                    data_train[id_] = data[id_]
                    labels_train[id_] = labels[id_]
            else:
                if cnt_false < 25:
                    cnt_false += 1
                    ids_val.append(id_)
                    data_val[id_] = data[id_]
                    labels_val[id_] = labels[id_]
                else:
                    ids_train.append(id_)
                    data_train[id_] = data[id_]
                    labels_train[id_] = labels[id_]
        # get test set from test_index
        ids_test, data_test, labels_test = [], {}, {}
        for index in test_index:
            id_ = ids[index]
            ids_test.append(id_)
            data_test[id_] = data[id_]
            labels_test[id_] = labels[id_]
        train = (ids_train, data_train, labels_train)
        val = (ids_val, data_val, labels_val)
        test = (ids_test, data_test, labels_test)

        # prepare by article cross validation data
        if constant.aug_count != '':
            data_loader_train, data_loader_val, data_loader_test, ids_val_dict, ids_test_dict = prepare_byarticle_cross_validation(train, val, test, constant.batch_size, constant.aug_count)
        else:
            data_loader_train, data_loader_val, data_loader_test = prepare_byarticle_cross_validation(train, val, test, constant.batch_size, constant.aug_count)

        # need to init the final Classifier for each fold
        if constant.use_bert:
            if constant.use_bert_plus_lstm:
                Classifier = models.Classifier(hidden_dim1=constant.hidden_dim, hidden_dim2=constant.hidden_dim_tit)
                # Classifier.load_state_dict(torch.load("bert_model/classifier_bypublisher2.bin"))
            else:
                Classifier = models.Classifier(hidden_dim1=768, hidden_dim2=768)
        else:
            Classifier = models.Classifier(hidden_dim1=constant.hidden_dim, hidden_dim2=constant.hidden_dim_tit)

        if constant.USE_CUDA:
            if constant.use_bert_plus_lstm:
                lstm_article.cuda()
                lstm_title.cuda()
            article_model.cuda()
            title_model.cuda()
            Classifier.cuda()

        criterion = nn.BCELoss()

        if constant.optimizer=='adam':
            opt = torch.optim.Adam(Classifier.parameters(), lr=constant.lr_classi, weight_decay=constant.weight_decay)
        elif constant.optimizer=='adagrad':
            opt = torch.optim.Adagrad(Classifier.parameters(), lr=constant.lr_classi)
        elif constant.optimizer=='sgd':
            opt = torch.optim.SGD(Classifier.parameters(), lr=constant.lr_classi, momentum=0.9)
        
        # set lr scheduler
        # scheduler = StepLR(opt, step_size=1, gamma=0.8)
        
        # set tensorboard folder name
        if constant.use_bert:
            experiment_name = "BERT_FineTune_aug{0}_LRlr{1}_k{2}".format(constant.aug_count, constant.lr_classi, k)
        else:
            experiment_name = "LSTM_FineTune_aug{0}_LRlr{1}_k{2}".format(constant.aug_count, constant.lr_classi, k)
        
        logdir = "tensorboard/" + experiment_name + "/"
        writer = SummaryWriter(logdir)
        global_steps = 0
        best_val_acc = 0
        # training and testifng
        for e in range(constant.max_epochs):
            # scheduler.step()
            article_model.train()
            title_model.train()
            Classifier.train()
            if constant.use_bert_plus_lstm:
                lstm_article.train()
                lstm_title.train()
            loss_log = []
            f1_log = 0
            acc_log = 0
            # training
            pbar = tqdm(enumerate(data_loader_train),total=len(data_loader_train))
            for i, (X, x_len, tit, tit_len, y, ind) in pbar:
                opt.zero_grad()
                if constant.use_bert:
                    X = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(item)) for item in X]
                    tit = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(item)) for item in tit]
                    X, segments_ids_article, tit, segments_ids_tit = padding_for_bert(X, tit)
                    if constant.USE_CUDA:
                        X, segments_ids_article, tit, segments_ids_tit, y = X.cuda(), segments_ids_article.cuda(), tit.cuda(), segments_ids_tit.cuda(), y.cuda()
                    encoded_article_layers, _ = article_model(X, segments_ids_article)
                    encoded_tit_layers, _ = title_model(tit, segments_ids_tit)
                    if constant.use_bert_plus_lstm:
                        _, article_hidden = lstm_article(encoded_article_layers[-1])
                        _, title_hidden = lstm_title(encoded_tit_layers[-1])
                        article_feat = article_hidden[-1][-1]
                        title_feat = title_hidden[-1][-1]
                    else:
                        article_feat = torch.sum(encoded_article_layers[-1], dim=1)
                        title_feat = torch.sum(encoded_tit_layers[-1], dim=1) #[batch_size, hidden_size]
                else:
                    article_feat = article_model.feature(X, x_len)
                    title_feat = title_model.feature(tit, tit_len)
                feature = torch.cat((article_feat, title_feat), dim=1)
                pred_prob = Classifier(feature)
                
                loss = criterion(pred_prob, y)
                loss.backward()
                opt.step()

                loss_log.append(loss.item())
                accuracy, microPrecision, microRecall, microF1 = getMetrics(pred_prob.detach().cpu().numpy(), y.cpu().numpy())
                f1_log += microF1
                acc_log += accuracy
                pbar.set_description("(Epoch {}) TRAIN F1:{:.4f} TRAIN LOSS:{:.4f} ACCURACY:{:.4f}".format((e+1), f1_log/float(i+1), np.mean(loss_log), acc_log/float(i+1)))

                writer.add_scalars('train', {'loss': np.mean(loss_log),
                                            'acc': acc_log/float(i+1),
                                            'f1': f1_log/float(i+1)}, global_steps)
                global_steps+=1
            
            """
                validate and test
                1. Get the test accuracy result from the model that gets the best accuracy in validation
                2. Whenever we find better accuracy result in the validation set, we need to test the model in the test 
                set and get the updated test set accuracy result.
                3. No need to save model during cross validation (cross validation is to find the best model)
            """
            article_model.eval()
            title_model.eval()
            Classifier.eval()
            if constant.use_bert_plus_lstm:
                lstm_article.eval()
                lstm_title.eval()
            print("Evaluation on validation set")
            use_add_feature_flag = constant.use_emo2vec_feat or constant.use_url
            if constant.use_bert:
                if constant.aug_count != '':
                    accuracy, pred, id_ = eval_bert_with_chunked_data(article_model, title_model, Classifier, data_loader_val, tokenizer, ids_val_dict, None, writer, e, False)
                else:
                    if constant.use_bert_plus_lstm:
                        accuracy, pred, id_ = eval_bert(article_model, title_model, Classifier, data_loader_val, tokenizer, lstm_article, lstm_title, use_add_feature_flag, writer, e, False)
                    else:
                        accuracy, pred, id_ = eval_bert(article_model, title_model, Classifier, data_loader_val, tokenizer, None, None, use_add_feature_flag, writer, e, False)
            else:
                accuracy, pred, id_ = eval_tit_lstm(article_model, title_model, Classifier, data_loader_val, use_add_feature_flag, writer, e, False)
            
            # find better accuracy in the validation set, need to test the model in the testset
            if(accuracy > best_val_acc):
                print("Find better model, test it on test set")
                best_val_acc = accuracy
                if constant.use_bert:
                    if constant.aug_count != '':
                        accuracy, pred, id_ = eval_bert_with_chunked_data(article_model, title_model, Classifier, data_loader_test, tokenizer, ids_test_dict, None, writer, e, True)
                    else:
                        if constant.use_bert_plus_lstm:
                            accuracy, pred, id_ = eval_bert(article_model, title_model, Classifier, data_loader_test, tokenizer, lstm_article, lstm_title, use_add_feature_flag, writer, e, True)
                        else:
                            accuracy, pred, id_ = eval_bert(article_model, title_model, Classifier, data_loader_test, tokenizer, None, None, use_add_feature_flag, writer, e, True)
                else:
                    accuracy, pred, id_ = eval_tit_lstm(article_model, title_model, Classifier, data_loader_test, use_add_feature_flag, writer, e, True)
                test_acc = accuracy
                if best_val_acc + test_acc > 1.53:
                    torch.save(Classifier.state_dict(), "bert_model/classifier.bin")
                    print("Classifier has been saved in bert_model/classifier.bin")
        # finish one fold, need to accumulate the test_acc (will do average of accuracy after k folds)
        avg_test_acc += test_acc
    
    # after k folds cross validation, get the final average test accuracy
    avg_test_acc = avg_test_acc * 1.0 / kfold
    print("After {0} folds cross validation, the final accuracy of {1} is {2}".format(kfold, constant.manual_name, avg_test_acc))


if __name__ == "__main__":

    if constant.do_train:
        train(aug_count=constant.aug_count)
    elif constant.do_eval_bert_plus_lstm:
        eval_bert_plus_lstm()
    elif constant.do_cross_validation:
        cross_validation()
    elif constant.do_predict:
        predict()
