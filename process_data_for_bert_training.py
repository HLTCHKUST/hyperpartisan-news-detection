
import pickle
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
import numpy as np

with open('/home/nayeon/fakenews/data_new/ids.pickle', "rb") as pkl_in:
    by_publisher_ids = pickle.load(pkl_in)
with open('/home/zihan/fakenews/data_new/preprocessed_new_train_clean.pickle', "rb") as pkl_in:
    by_publisher_train = pickle.load(pkl_in)
with open('/home/zihan/fakenews/data_new/preprocessed_new_val_clean.pickle', "rb") as pkl_in:
    by_publisher_val = pickle.load(pkl_in)

article_corpus = open("./data_new/article_corpus.txt", "w")

# by publisher train
ids_train = by_publisher_ids["train"]
np.random.shuffle(ids_train)
for id_ in tqdm(ids_train):
    article = by_publisher_train[id_]["article_text"]
    article = article.strip()
    article2sentlist = sent_tokenize(article)
    if len(article2sentlist) >= 20:
        for sent in article2sentlist[2:20]:
            sent = sent.strip()
            if len(sent) <= 1: continue
            article_corpus.write(sent+"\n")
    else:
        for sent in article2sentlist:
            sent = sent.strip()
            if len(sent) <= 1: continue
            article_corpus.write(sent+"\n")
    # need "\n" to split diff articles
    article_corpus.write("\n")
# by publisher val
ids_val = by_publisher_ids["val"]
np.random.shuffle(ids_val)
for id_ in tqdm(ids_val):
    article = by_publisher_val[id_]["article_text"]
    article = article.strip()
    article2sentlist = sent_tokenize(article)
    if len(article2sentlist) >= 20:
        for sent in article2sentlist[2:20]:
            sent = sent.strip()
            if len(sent) <= 1: continue
            article_corpus.write(sent+"\n")
    else:
        for sent in article2sentlist:
            sent = sent.strip()
            if len(sent) <= 1: continue
            article_corpus.write(sent+"\n")
    # need "\n" to split diff articles
    article_corpus.write("\n")

article_corpus.close()