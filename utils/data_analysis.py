
import pickle
from nltk.tokenize import word_tokenize
import numpy as np
from tqdm import tqdm

def article_length(articles, data_type):
    print ("analysing", data_type)
    max_len = 0
    total_len = 0
    min_len = 100000
    for article in tqdm(articles):
        word_tokens = word_tokenize(article)
        article_len = len(word_tokens)
        if article_len < 20:
            print (article)
        # print (article_len)
        if article_len > max_len:
            max_len = article_len
        if article_len < min_len:
            min_len = article_len
        total_len += article_len
    avg_len = total_len / len(articles)
    print ("maximum length:", max_len, "average length:", avg_len, "minimum length:", min_len)

def article_length2(pickle_val, ids_val):
    for i in tqdm(range(0, len(ids_val))):
        id_ = ids_val[i]
        article_text = pickle_val[id_]["article_text"]
        word_tokens = word_tokenize(article_text)
        article_len = len(word_tokens)
        if article_len < 30:
            print ("length of the articles:", article_len)
            print ("id:", id_)
            print ("article:", article_text)

def internal_length(articles_internal_link, data_type):
    print ("analysing", data_type, "internal")
    max_internal = 0 
    total_internal = 0
    min_internal = 100000
    for inter_num in tqdm(articles_internal_link):
        if inter_num > max_internal:
            max_internal = inter_num
        if inter_num < min_internal:
            min_internal = inter_num
        total_internal += inter_num
    avg_internal = total_internal * 1.0 / len(articles_internal_link)
    print ("maximum internal:", max_internal, "average internal:", avg_internal, "minimum internal:", min_internal)

def external_length(articles_external_link, data_type):
    print ("analysing", data_type, "external")
    max_external = 0 
    total_external = 0
    min_external = 100000
    for item in tqdm(articles_external_link):
        exter_num = len(item)
        if exter_num > max_external:
            max_external = exter_num
        if exter_num < min_external:
            min_external = exter_num
        total_external += exter_num
    avg_external = total_external * 1.0 / len(articles_external_link)
    print ("maximum external:", max_external, "average external:", avg_external, "minimum external:", min_external)

def count_bias_extent(label_dict, ids):
    cnt_bias_extent_dict = {"right": 0, "right-center": 0, "left": 0, "left-center": 0, "least": 0}
    for i in tqdm(range(0, len(ids))):
        id_ = ids[i]
        # print (label_dict[id_]['bias_extent'])
        bias = label_dict[id_]['bias_extent']
        # if bias != "least" and label_dict[id_]["label"] == "false":
        #     print ("bias:", bias, "label:", label_dict[id_]["label"])
        cnt_bias_extent_dict[bias] += 1
    print (cnt_bias_extent_dict)

# get articles
# with open('/home/zihan/fakenews/data_new/preprocessed_new_train.pickle', 'rb') as handle:
#     print ("loading preprocessed_new_train.pickle ...")
#     pickle_train = pickle.load(handle)
# with open('/home/zihan/fakenews/data_new/preprocessed_new_val.pickle', 'rb') as handle:
#     print ("loading preprocessed_new_val.pickle ...")
#     pickle_val = pickle.load(handle)
with open('/home/nayeon/fakenews/data_new/new_ids.pickle', 'rb') as handle:
    print ("loading new_ids.pickle ...")
    ids = pickle.load(handle)
ids_train = ids['train']
ids_val = ids['val']

# articles_train = []
# train_internal_link = []
# train_external_link = []
# for i in range(0, len(ids_train)):
#     id_ = ids_train[i]
#     articles_train.append(pickle_train[id_]["article_text"])
#     train_internal_link.append(pickle_train[id_]["internal"])
#     train_external_link.append(pickle_train[id_]["external"])
# article_length(articles_train, "training set")
# internal_length(train_internal_link, "training set")
# external_length(train_external_link, "training set")

# articles_val = []
# val_internal_link = []
# val_external_link = []
# for i in range(0, len(ids_val)):
#     id_ = ids_val[i]
#     articles_val.append(pickle_val[id_]["article_text"])
#     val_internal_link.append(pickle_val[id_]["internal"])
#     val_external_link.append(pickle_val[id_]["external"])
# article_length(articles_val, "validation set")
# internal_length(val_internal_link, "validation set")
# external_length(val_external_link, "validation set")

with open('/home/nayeon/fakenews/data_new/train_labels.pickle', 'rb') as handle:
    print ("loading train_labels.pickle ...")
    train_labels = pickle.load(handle)
with open('/home/nayeon/fakenews/data_new/val_labels.pickle', 'rb') as handle:
    print ("loading val_labels.pickle ...")
    val_labels = pickle.load(handle)

# count_bias_extent(train_labels, ids_train)
# count_bias_extent(val_labels, ids_val)
cnt_train_true = 0
cnt_train_false = 0
cnt_val_true = 0
cnt_val_false = 0
for id_ in ids_train:
    if train_labels[id_]["label"] == "true":
        cnt_train_true += 1
    if train_labels[id_]["label"] == "false":
        cnt_train_false += 1
for id_ in ids_val:
    if val_labels[id_]["label"] == "true":
        cnt_val_true += 1
    if val_labels[id_]["label"] == "false":
        cnt_val_false += 1
print ("fake news in train:", cnt_train_true, "true news in train", cnt_train_false, "percentage of fake news:", cnt_train_true * 1.0 / (cnt_train_false+cnt_train_true))
print ("fake news in val:", cnt_val_true, "true news in val", cnt_val_false, "percentage of fake news:", cnt_val_true * 1.0 / (cnt_val_false+cnt_val_true))

# topic_num_dict = {}
# for key, value in topic_id_dict.items():
#     topic_num_dict[key] = 0
# print (topic_num_dict)

# # analyze the top 100K data in the training set
# fake_cnt = 0
# cnt_total = 0
# for i in tqdm(range(0, len(ids_train))):
#     id_ = ids_train[i]
#     for key, value in topic_id_dict.items():
#         if id_ in value:
#             topic_num_dict[key] += 1
#             cnt_total += 1
# #     label = train_labels[id_]["label"]
# #     if label == "true":
# #         fake_cnt += 1
# # print ("proportion of fake news", fake_cnt * 1.0 / 100000)
# for key, value in topic_num_dict.items():
#     topic_num_dict[key] = topic_num_dict[key] * 1.0 / cnt_total

# print (topic_num_dict)