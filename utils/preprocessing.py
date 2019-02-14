import argparse
import pickle
from gensim.corpora import Dictionary
from data_utils import parse_xml, clean_txt
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from tqdm import tqdm
import string
from bs4 import BeautifulSoup as Soup
import numpy as np

def new_data_cleaning():
    for phase in ['train','val']:
        preprocessed_dict = {}
        if phase == 'train':
            base_path = 'data_new/articles-training-bypublisher-20181122.xml'
        else:
            base_path = 'data_new/articles-validation-bypublisher-20181122.xml'
        
        print("Loading data from xml")
        xml_file = open(base_path).read()
        soup = Soup(xml_file)
        articles=soup.find_all('article')
        print("Number of articles: ", len(articles))

        for a in tqdm(articles):
            id, article = parse_xml(a)
            preprocessed_dict[id] = article

        with open('data_new/preprocessed_new_{}.pickle'.format(phase), 'wb') as handle:
            pickle.dump(preprocessed_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

def data_cleaning(ids):
    for phase in ['train','val']:
        preprocessed_dict = {}
        if phase == 'train':
            base_path = 'data/articles-training'
        else:
            base_path = 'data/articles-validation'

        for i, id_ in enumerate(tqdm(ids[phase])):
            f_name = '{}/{}.xml'.format(base_path,id_)
            id, article = parse_xml(f_name)
            preprocessed_dict[id] = article

        with open('data/preprocessed_cleaner_{}.pickle'.format(phase), 'wb') as handle:
            pickle.dump(preprocessed_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

def dic_creation(ids):
    for phase in ['train','val']:
        print(phase)
        with open('data/preprocessed_{}.pickle'.format(phase), 'rb') as preprocessed:
            preprocessed_dict = pickle.load(preprocessed)
            article_txts = []
            for id_ in tqdm(ids[phase]):
                article_txts.append(preprocessed_dict[id_]['article_text'])
#             article_txts = list(map(lambda id_: preprocessed_dict[id_]['article_text'], ids[phase]))
            print("loaded text")
            article_tokens = []
            for txt in tqdm(article_txts):
                article_tokens.append(word_tokenize(txt))
#             article_tokens = list(map(lambda txt: word_tokenize(txt), article_txts))
            print("tokenized")
            
        if phase == 'train':
            dct = Dictionary(article_tokens)
        else:
            with open('data/dict.pickle', 'rb') as handle:
                dct = pickle.load(handle)
                dct.add_documents(article_tokens)
        print("dict processed")
        
        # save dict
        with open('data/dict.pickle', 'wb') as handle:
            pickle.dump(dct, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print("saved")

def get_badword_list():
    punc_list = list(map(lambda t: t, string.punctuation))
    bad_word_list = set(stopwords.words("english")+['said','say','people','numberplaceholder',"\'s","\'m","...","www","http"]+punc_list)
    return bad_word_list

def find_bad_ids(dct, bad_word_list):
    bad_ids=[]
    for w in bad_word_list:
        try:
            id_ = dct.token2id[w]
            bad_ids.append(id_)
        except KeyError:
            continue
            print ('I got a KeyError - reason')
    return bad_ids

def further_cleaning2doc2bow_lda(phase,ids,dct): 
    print("start further cleaning 2 doc2bow")
    with open('data/preprocessed_{}.pickle'.format(phase), 'rb') as preprocessed:
        preprocessed_data = pickle.load(preprocessed)
    
    bad_word_list = get_badword_list()
    bad_ids = find_bad_ids(dct,bad_word_list)
    dct.filter_tokens(bad_ids=bad_ids)
    
    corpus=[]
    for i, id_ in enumerate(tqdm(ids[phase])):
        text = preprocessed_data[id_]['article_text']       
        text = text.split()
        text = [w for w in text if w not in bad_word_list]
        text = " ".join(text)
        
        corpus.append(dct.doc2bow(word_tokenize(text)))
        
    with open('data/further_cleaned_doc2bow_{}.pickle'.format(phase), 'wb') as handle:
        pickle.dump(corpus, handle, protocol=pickle.HIGHEST_PROTOCOL)

def run_doc2bow(phase,ids,dct):   
    with open('data/preprocessed_{}.pickle'.format(phase), 'rb') as preprocessed:
        preprocessed_data = pickle.load(preprocessed)
    
    # filter bad, useless ids from dct
    bad_ids = find_bad_ids()
    dct.filter_tokens(bad_ids=bad_ids)
    
    train_corpus=[]
    for i, id_ in enumerate(tqdm(ids[phase])):
        text = preprocessed_data[id_]['article_text']
        train_corpus.append(dct.doc2bow(word_tokenize(text)))
            
    with open('data/doc2bow_{}.pickle'.format(phase), 'wb') as handle:
        pickle.dump(train_corpus, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
def run_tokenize(phase, ids):   
    in_template = 'data/preprocessed_{}.pickle'
    out_template = 'data/tokenized_{}.pickle'
    with open(in_template.format(phase), 'rb') as preprocessed:
        preprocessed_data = pickle.load(preprocessed)
        
    for i, id_ in enumerate(tqdm(ids[phase])):
        text = preprocessed_data[id_]['article_text']
        preprocessed_data[id_]['article_text'] = word_tokenize(text)

    with open(out_template.format(phase), 'wb') as handle:
        pickle.dump(preprocessed_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def data_preprocess(train_data_path, val_data_path, ids_path):
    print("data preprocessing ...")
    pickle_train = open(train_data_path, 'rb')
    pickle_val = open(val_data_path, 'rb')
    pickle_ids = open(ids_path, 'rb')
    train_data_without_clean = pickle.load(pickle_train)
    val_data_without_clean = pickle.load(pickle_val)
    ids = pickle.load(pickle_ids)
    pickle_train.close()
    pickle_val.close()
    pickle_ids.close()
    train_data_clean = {}
    val_data_clean = {}
    for id_ in tqdm(ids['train']):
        title = train_data_without_clean[id_]["title"]
        article = train_data_without_clean[id_]["article_text"]
        article_clean = clean_txt(article, remove_stopwords=False)
        train_data_clean[id_] = {"article_text": article_clean, "title": title}
    for id_ in tqdm(ids['val']):
        title = val_data_without_clean[id_]["title"]
        article = val_data_without_clean[id_]["article_text"]
        article_clean = clean_txt(article, remove_stopwords=False)
        val_data_clean[id_] = {"article_text": article_clean, "title": title}
    pickle_train_out = open("/home/zihan/fakenews/data_new/preprocessed_new_train_clean.pickle", "wb")
    pickle_val_out = open("/home/zihan/fakenews/data_new/preprocessed_new_val_clean.pickle", "wb")
    print("dumping pickle file ...")
    pickle.dump(train_data_clean, pickle_train_out, protocol=4)
    pickle.dump(val_data_clean, pickle_val_out, protocol=4)

def reorder_data(data, info):
    print("reodering", info, "data in terms of the number of sentences ...")
    ids, X, labels = data
    lengths = []
    print("getting the sentence lengths of articles ...")
    for id_ in tqdm(ids):
        sents = sent_tokenize(X[id_]["article_text"])
        lengths.append(len(sents))
    # get index
    print("sorting ...")
    index = sorted(range(len(lengths)), key=lambda k: lengths[k])
    ids = np.array(ids)
    ids = ids[index]
    
    return (ids, X, labels)
