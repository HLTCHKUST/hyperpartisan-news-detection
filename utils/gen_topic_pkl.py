from gensim import models
from nltk.tokenize import word_tokenize

import pickle
import numpy as np
from tqdm import tqdm

# with open('/home/zihan/fakenews/data_new/preprocessed_new_train.pickle', 'rb') as handle:
#     print ("loading preprocessed_new_train.pickle ...")
#     pickle_train = pickle.load(handle)
# with open('/home/zihan/fakenews/data_new/preprocessed_new_val.pickle', 'rb') as handle:
    # print ("loading preprocessed_val.pickle ...")
    # pickle_valid = pickle.load(handle)
with open('/home/nayeon/fakenews/data_new/new_ids.pickle', 'rb') as handle:
    print ("loading ids.pickle ...")
    ids = pickle.load(handle)
    # ids_train = ids['train']
    ids_valid = ids['val']

# articles_train = []
# for i in range(0, len(ids_train)):
#     id_ = ids_train[i]
#     articles_train.append(pickle_train[id_]["article_text"])

# articles_val = []
# for i in range(0, len(ids_valid)):
#     id_ = ids_valid[i]
#     articles_val.append(pickle_valid[id_]["article_text"])

# with open("/home/nayeon/fakenews/data/dict_wrong.pickle", 'rb') as handle:
#     dictionary = pickle.load(handle)
# corpus = [dictionary.doc2bow(word_tokenize(text)) for text in tqdm(articles_val)]

# print ("dumping old doc2bow corpus ...")
# pkl_out = open('../data_new/old_doc2bow_corpus_val.pickle', 'wb')
# pickle.dump(corpus, pkl_out, protocol=4)
# print ("finish dumping!")

with open("../data_new/old_doc2bow_corpus_val.pickle", 'rb') as handle:
    print ("loading old_doc2bow_corpus_val.pickle")
    corpus = pickle.load(handle)
    print("finish loading old_doc2bow_corpus_val.pickle")

# [15,20,25,30,40,50,75]
# model_name_list = ["lda_nt_75"]
model_name_list = ["lda_nb0_nt_20", "lda_nb0_nt_30"]
# model_name_list = ["lda_nt_15", "lda_nt_20", "lda_nt_25", "lda_nt_30"]
# model_name_list = ["lda_nt_40", "lda_nt_50", "lda_nt_75"]

for model_name in model_name_list:
    print ("parsing", model_name, "...")
    # path = "/home/nayeon/fakenews/trained/lda/"+model_name+".model"
    path = "/home/nayeon/fakenews/trained_lda_old/"+model_name+".model"
    lda_model = models.LdaModel.load(path)
    # print (lda_model)
    id_topicProb = {}
    topic_id_top1 = {}
    topic_id_top3 = {}
    for i, doc in tqdm(enumerate(corpus)):
        # id_ = ids_train[i]
        id_ = ids_valid[i]
        # print ("id:", id_)
        vector = lda_model[doc]
        vector.sort(key=lambda tup: tup[1], reverse=True)
        # print (vector)
        id_topicProb[id_] = vector

        vector_top1 = vector[0:1]
        # choose top3 topics
        if len(vector) > 2:
            vector_top3 = vector
        else:
            vector_top3 = vector[0:3]

        for tup in vector_top1:
            topic_num = tup[0]
            key = "topic"+str(topic_num)
            if key not in topic_id_top1:
                topic_id_top1[key] = [id_]
            else:
                topic_id_top1[key].append(id_)

        for tup in vector_top3:
            topic_num = tup[0]
            key = "topic"+str(topic_num)
            if key not in topic_id_top3:
                topic_id_top3[key] = [id_]
            else:
                topic_id_top3[key].append(id_)
    # print ("id_topicProb")
    # print (id_topicProb)
    # print ("topic_id_top3")
    # print (topic_id_top3)
    # print ("topic_id_top1")
    # print (topic_id_top1)

    print ("dumping", model_name, "old_id_topicProb_val")
    pkl_out = open('../data_new/'+model_name+'_old_id_topicProb_val.pickle', 'wb')
    pickle.dump(id_topicProb, pkl_out, protocol=4)

    print ("dumping", model_name, "old_topic_id_val_top3")
    pkl_out = open('../data_new/'+model_name+'_old_topic_id_val_top3.pickle', 'wb')
    pickle.dump(topic_id_top3, pkl_out, protocol=4)

    print ("dumping", model_name, "old_topic_id_val_top1")
    pkl_out = open('../data_new/'+model_name+'_old_topic_id_val_top1.pickle', 'wb')
    pickle.dump(topic_id_top1, pkl_out, protocol=4)

topic_num = 20
with open("../data_new/lda_nb0_nt_20_old_id_topicProb_val.pickle", "rb") as handle:
    id_topicProb = pickle.load(handle)
# print (id_topicProb)
with open('/home/nayeon/fakenews/data_new/new_ids.pickle', 'rb') as handle:
    print ("loading ids.pickle ...")
    ids = pickle.load(handle)
# ids_train = ids['train']
ids_val = ids["val"]
id_allTopicProb = {}
for i in tqdm(range(0, len(ids_val))):
    prob_distri = np.zeros(topic_num)
    id_ = ids_val[i]
    for tup in id_topicProb[id_]:
        topic = tup[0]
        prob = tup[1]
        prob_distri[topic] = prob
    id_allTopicProb[id_] = prob_distri

# print ("id_20TopicProb")
# print (id_allTopicProb)
print ("dumping id_20TopicProb_val.pickle")
pkl_out = open("../data_new/id_20TopicProb_val.pickle", "wb")
pickle.dump(id_allTopicProb, pkl_out, protocol=4)