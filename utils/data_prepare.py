import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
from utils import constant 

def generate_val_set(ids, xs, labels, val_ratio):
    train_ids, train_x, train_label = [], {}, {}
    val_ids, val_x, val_label  = [], {}, {}
    cnt_true, cnt_false = 0, 0
    val_count = len(xs)*val_ratio*0.5 # binary thus x 0.5
    
    # shuffle ids
    np.random.shuffle(ids)
    for i in ids:
        x = xs[i]
        label = labels[i]["label"]
        if label == "true" and cnt_true < val_count:
            cnt_true += 1
            val_ids.append(i)
            val_x[i]=x 
            val_label[i]=label
        elif label == "false" and cnt_false < val_count:
            cnt_false += 1
            val_ids.append(i)
            val_x[i]=x 
            val_label[i]=label
        else:           
            train_ids.append(i)
            train_x[i]=x 
            train_label[i]=label
    
    return (train_ids,train_x,train_label), (val_ids,val_x,val_label)

def read_data(x_path, debug_mode=False, is_shuffle=False, random_state=0):
    # TODO implement shuffle
    paths = [
        '/home/nayeon/fakenews/data_new/ids',
        '/home/nayeon/fakenews/data_new/train_labels',
        x_path.format('train'),
        '/home/nayeon/fakenews/data_new/val_labels',
        x_path.format('val')
    ]

    if debug_mode:
        paths = list(map(lambda p:p+'_1000',paths))
    paths = list(map(lambda p:p+'.pickle',paths))
        
    with open(paths[0], 'rb') as handle:
        ids = pickle.load(handle)
    with open(paths[1], 'rb') as handle:
        labels = pickle.load(handle)
    with open(paths[2],'rb') as handle:
        xs = pickle.load(handle)
        xs_ids = ids['train']
           
    train, val = generate_val_set(xs_ids, xs, labels, 0.2)

    print("the length of train set:", len(train[0]))
    print("the length of val set:", len(val[0]))
    
    # prepare "testset" - thus, no need labels. currently, just include test labels
    with open(paths[4],'rb') as handle:
        test_x = pickle.load(handle)
        test_ids = ids['val']
        print("the length of test set:", len(test_ids))
        
    with open(paths[3], 'rb') as handle:
        t_labels = pickle.load(handle)
        test_labels={}
        for i in test_ids:
            test_labels[i] = t_labels[i]['label']

    # For external_url features - build dict from only trainset
    url_dic_path = '/home/nayeon/fakenews/data_new/url_dic.pickle'
    if Path(url_dic_path).is_file():
        with open(url_dic_path, 'rb') as handle:
            print("loading url_dic from {}".format(url_dic_path))
            url_dic = pickle.load(handle)
#             print(url_dic.url2index['twitter'])
#             print(url_dic.urls_to_idx_seq(["http://www.twitter.com"]))
    else:
        url_dic = ExUrl()
        with open("/home/nayeon/fakenews/data_new/train_idExternal.pickle", "rb") as f_in:
            train_idExternal = pickle.load(f_in)
            for urls in train_idExternal.values():
                url_dic.index_urls(list(set(urls)))
            url_dic.trim()
        if not constant.debug_mode:
            with open(url_dic_path, 'wb') as handle:
                pickle.dump(url_dic, handle, pickle.HIGHEST_PROTOCOL)
                print("save url dic")
        # with open("./data_new/val_idExternal.pickle", "rb") as f_in:
        #     val_idExternal = pickle.load(f_in)

    return train, val, (test_ids, test_x, test_labels), url_dic
#     return train, val, (test_ids, test_x, None), vocab


def read_byarticle_data(test_whole=False):
    if test_whole == True:
        print("loading the whole by article data ...")
        ids_file = open("data_new/by_article_ids.pickle", "rb")
        ids = pickle.load(ids_file)
        data_file = open("data_new/preprocessed_byarticle_data.pickle", "rb")
        data = pickle.load(data_file)
        labels_file = open("data_new/by_article_labels.pickle", "rb")
        labels  = pickle.load(labels_file)
        
        ids2, data2, labels2 = [], {}, {}
        cnt_true, cnt_false = 0, 0
        # shuffle and get 238 true and 238 false
        # np.random.shuffle(ids)
        for id_ in ids:
            if labels[id_] == "true":
                if cnt_true < 238:
                    cnt_true += 1
                    ids2.append(id_)
                    data2[id_] = data[id_]
                    labels2[id_] = labels[id_]
            else:
                if cnt_false < 238:
                    cnt_false += 1
                    ids2.append(id_)
                    data2[id_] = data[id_]
                    labels2[id_] = labels[id_]
        return (ids2, data2, labels2)
        # return (ids, data, labels)

    else:
        ids_file = open("data_new/by_article_ids_train.pickle", "rb")
        ids_train = pickle.load(ids_file)
        data_file = open("data_new/preprocessed_byarticle_train.pickle", "rb")
        data_train = pickle.load(data_file)
        labels_file = open("data_new/by_article_labels_train.pickle", "rb")
        labels_train = pickle.load(labels_file)

        ids_file = open("data_new/by_article_ids_test.pickle", "rb")
        ids_test = pickle.load(ids_file)
        data_file = open("data_new/preprocessed_byarticle_test.pickle", "rb")
        data_test = pickle.load(data_file)
        labels_file = open("data_new/by_article_labels_test.pickle", "rb")
        labels_test = pickle.load(labels_file)

        return (ids_train, data_train, labels_train), (ids_test, data_test, labels_test)

def read_filtered_data(filter=True):
    if filter == False:
        print("loading whole ids (without filtering)")
        with open("/home/nayeon/fakenews/data_new/ids.pickle", "rb") as ids_file:
            ids = pickle.load(ids_file)
        train_ids = ids["train"]
        test_ids = ids["val"]
    else:
        with open("data_new/filtered_bypublisher_train_ids_full.pickle", "rb") as train_ids_file:
            train_ids = pickle.load(train_ids_file)
        with open("data_new/filtered_bypublisher_val_ids.pickle", "rb") as val_ids_file:
            test_ids = pickle.load(val_ids_file)
    with open("/home/nayeon/fakenews/data_new/preprocessed_new_train_wtitle.pickle", "rb") as train_data_file:
        tr_data = pickle.load(train_data_file)
    with open("/home/nayeon/fakenews/data_new/preprocessed_new_val_wtitle.pickle", "rb") as val_data_file:
        v_data = pickle.load(val_data_file)
    with open('/home/nayeon/fakenews/data_new/train_labels.pickle', 'rb') as train_labels_file:
        tr_labels = pickle.load(train_labels_file)
    with open('/home/nayeon/fakenews/data_new/val_labels.pickle', 'rb') as val_labels_file:
        v_labels = pickle.load(val_labels_file)
    
    train_data, test_data, train_labels, test_labels = {}, {}, {}, {}

    # split filtered_ids it into train and test
    for id_ in train_ids:
        train_data[id_] = tr_data[id_]
        train_labels[id_] = tr_labels[id_]["label"]
    for id_ in test_ids:
        test_data[id_] = v_data[id_]
        test_labels[id_] = v_labels[id_]["label"]
    filtered_train = (train_ids, train_data, train_labels)
    filtered_test = (test_ids, test_data, test_labels)
    
    return filtered_train, filtered_test

def read_byarticle_aug_data(aug_count):
    # original split ids
    with open("data_new/by_article_ids_train.pickle", "rb") as handle:
        org_ids_train = pickle.load(handle)
    with open("data_new/by_article_ids_test.pickle", "rb") as handle:
        org_ids_test = pickle.load(handle)
    with open("data_new/preprocessed_byarticle_test.pickle", "rb") as handle:
        org_data_test = pickle.load(handle)
    with open("data_new/by_article_labels_test.pickle", "rb") as handle:
        org_labels_test = pickle.load(handle)

    # load augmented data-before-split
    with open("data_new/by_article_augmented_{0}.pickle".format(aug_count), "rb") as handle:
        by_article_aug = pickle.load(handle)
        
    ids_train, ids_test, data_train, data_test, labels_train, labels_test = [],[],{},{},{},{}
    cnt_true, cnt_false = 0, 0
    for k in by_article_aug:
        if by_article_aug[k]['id'] in org_ids_train:
            ids_train.append(k)
            data_train[k]=by_article_aug[k]
            labels_train[k]=by_article_aug[k]['label']
            # print(by_article_aug[k]["label"])
            if by_article_aug[k]["label"] == "true":
                cnt_true += 1
            else:  
                cnt_false += 1
        # else:
        #     ids_test.append(k)
        #     data_test[k]=by_article_aug[k]
        #     labels_test[k]=by_article_aug[k]['label']
    # print(len(by_article_aug))
    print("length of train set and test set")
    print(len(ids_train), len(org_ids_test))
    print("length of true and false in training set")
    print(cnt_true, cnt_false)
    return (ids_train, data_train, labels_train), (org_ids_test, org_data_test, org_labels_test)
    # return (ids_train, data_train, labels_train), (ids_test, data_test, labels_test)

def read_byarticle_aug_data_for_cross_validation(train, val, test, aug_count):
    # load augmented data-before-split
    with open("data_new/by_article_augmented_{0}.pickle".format(aug_count), "rb") as handle:
        by_article_aug = pickle.load(handle)
    
    ids_train, data_train, labels_train = train
    ids_val, data_val, labels_val = val
    ids_test, data_test, labels_test = test
    ids_val_dict, ids_test_dict = {}, {}

    chunked_ids_train, chunked_data_train, chunked_labels_train = [], {}, {}
    chunked_ids_val, chunked_data_val, chunked_labels_val = [], {}, {}
    chunked_ids_test, chunked_data_test, chunked_labels_test = [], {}, {}
    for k in by_article_aug:
        if by_article_aug[k]['id'] in ids_train:
            chunked_ids_train.append(k)
            chunked_data_train[k]=by_article_aug[k]
            chunked_labels_train[k]=by_article_aug[k]['label']
        elif by_article_aug[k]['id'] in ids_val:
            chunked_ids_val.append(k)
            chunked_data_val[k]=by_article_aug[k]
            chunked_labels_val[k]=by_article_aug[k]['label']
            if by_article_aug[k]['id'] not in ids_val_dict:
                ids_val_dict[by_article_aug[k]['id']] = [k]
            else:
                ids_val_dict[by_article_aug[k]['id']].append(k)
        else:
            chunked_ids_test.append(k)
            chunked_data_test[k]=by_article_aug[k]
            chunked_labels_test[k]=by_article_aug[k]['label']
            if by_article_aug[k]['id'] not in ids_test_dict:
                ids_test_dict[by_article_aug[k]['id']] = [k]
            else:
                ids_test_dict[by_article_aug[k]['id']].append(k)
    print("length of train set, val set and test set")
    print(len(ids_train), len(ids_val), len(ids_test))
    print("length of chunked train set, chunked val set and chunked test set")
    print(len(chunked_ids_train), len(chunked_ids_val), len(chunked_ids_test))
    
    chunked_train = (chunked_ids_train, chunked_data_train, chunked_labels_train)
    chunked_val = (chunked_ids_val, chunked_data_val, chunked_labels_val)
    chunked_test = (chunked_ids_test, chunked_data_test, chunked_labels_test)

    return chunked_train, chunked_val, chunked_test, ids_val_dict, ids_test_dict