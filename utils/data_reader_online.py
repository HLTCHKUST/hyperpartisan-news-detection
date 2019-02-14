import pickle
import math
import torch
from torch.utils import data
from torch.autograd import Variable
import numpy as np
from nltk.tokenize import word_tokenize
from pathlib import Path
from tqdm import tqdm
from utils import constant
# from utils.input_reduction import InputReducer
from utils.data_utils import parse_xml, clean_txt
from utils.feature_utils import EmoFeatures
from utils.data_prepare import *
import validators
import re
from urllib.parse import urlparse
import string

def extract_base_url(url):
    pattern = re.compile("www")

    if validators.url(url):
        parsed = urlparse(url)
        base_url = parsed.netloc
        base_url = base_url.split(".")[0] if pattern.match(base_url) == None else base_url.split(".")[1]
        return base_url
    else:
        # invalid url
        return "invalid_url"

def lineToChar1hot(str):
    all_letters = string.ascii_letters + " .,;'"
    n_letters = len(all_letters)
    tensor = torch.zeros(len(str)).long()
#     tensor = np.zeros(len(str))
    for c in range(len(str)):
        try:
            tensor[c] = all_letters.index(str[c])
        except:
            continue
    return tensor #Variable(tensor)

class ExUrl:
    def __init__(self):
        self.url2count = {}
        self.url2index = {"INVAL": constant.INVALID_idx}
        self.index2url = {constant.INVALID_idx: "INVAL"}
        self.n_urls = 1
    
    def __len__(self):
        return len(self.url2index)

    def index_urls(self, urls):
        for url in urls:
            self.index_url(url)

    def index_url(self, url):
        if url not in self.url2index:
            self.url2count[url] = 1
            self.url2index[url] = self.n_urls
            self.index2url[self.n_urls] = url
            self.n_urls += 1
        else:
            self.url2count[url] += 1
    
    def trim(self, min_frequency=2):
        idx = 1
        self.url2index = {"INVAL": constant.INVALID_idx}
        self.index2url = {constant.INVALID_idx: "INVAL"}
        
        for url, count in self.url2count.items():
            if count < min_frequency:
                continue
            self.url2index[url] = idx
            self.index2url[idx] = url
            idx += 1
        self.n_urls = idx
    
    def urls_to_idx_1_hot(self, urls):
        idx_1_hot = np.zeros(self.n_urls)
        if urls==None:
            return idx_1_hot
        
        pattern = re.compile("www")
        
        for url in urls:
            base_url = extract_base_url(url)
            if base_url == 'invalid_url':
                idx_1_hot[constant.INVALID_idx] = 1
            elif base_url in self.url2index:
                url_idx = self.url2index[base_url]
                idx_1_hot[url_idx] = 1

        return idx_1_hot

class MyTokenizer():
    def tokenize(self, sentence):
        tok = []
        special_cases = ["'s", "'ve", "n't", "'re", "'d", "'ll"]
        for word in word_tokenize(sentence.lower()):
            flag = False
            for case in special_cases:
                if case not in word:
                    continue
                idx = word.find(case)
                tok.append(word[:idx])
                tok.append(word[idx:])
                flag = True
                break
            if not flag:
                tok.append(word)
        return tok

class Lang:
    def __init__(self):
        self.word2count = {}
        self.word2index = {"UNK":constant.UNK_idx, "PAD":constant.PAD_idx}
        self.index2word = {constant.UNK_idx: "UNK", constant.PAD_idx: "PAD"} 
        self.n_words = 2 # Count default tokens
        self.tt = MyTokenizer()
    
    def __len__(self):
        'Denotes the total number of samples'
        return self.n_words
    
    def tokenize(self, sentence):
        tok = []
        for word in word_tokenize(sentence):
            tok.append(word)
        return tok
        
    def index_words(self, sentence):
        for word in self.tokenize(sentence):
            self.index_word(word)

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
            
    def trim(self, min_frequency, max_frequency=-1):
        """Trims vocabulary for minimum frequency.
        Args:
          min_frequency: minimum frequency to keep.
          max_frequency: optional, maximum frequency to keep.
          Useful to remove very frequent categories (like stop words).
        """
        # Sort by alphabet then reversed frequency.
        idx = 2
        self.word2index = {"UNK":constant.UNK_idx, "PAD":constant.PAD_idx}
        self.index2word = {constant.UNK_idx: "UNK", constant.PAD_idx: "PAD"} 
        
        for word, count in self.word2count.items():
            if max_frequency > 0 and count >= max_frequency:
                continue
            if count <= min_frequency:
                continue
            self.word2index[word] = idx
            self.index2word[idx] = word
            idx += 1
        self.n_words = idx
        
    def keep_wanted_list(self, wanted_word_list):        
        idx = 2
        self.word2index = {"UNK":constant.UNK_idx, "PAD":constant.PAD_idx}
        self.index2word = {constant.UNK_idx: "UNK", constant.PAD_idx: "PAD"} 
        
        for word, count in self.word2count.items():
            if word in wanted_word_list:
                self.word2index[word] = idx
                self.index2word[idx] = word
                idx += 1
            else:
                continue
        self.n_words = idx
        
    def trim_unwanted_list(self, unwanted_list):        
        idx = 2
        self.word2index = {"UNK":constant.UNK_idx, "PAD":constant.PAD_idx}
        self.index2word = {constant.UNK_idx: "UNK", constant.PAD_idx: "PAD"} 
        
        for word, count in self.word2count.items():
            if word in unwanted_list:
                continue
            self.word2index[word] = idx
            self.index2word[idx] = word
            idx += 1
        self.n_words = idx

class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
    """

    def __init__(self, dataset, indices=None, num_samples=None):
        # if indices is not provided, 
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset.ids))) \
            if indices is None else indices
            
        # if num_samples is not provided, 
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples
            
        # distribution of classes in the dataset 
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, dataset.ids[idx])
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1
                
        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, dataset.ids[idx])] for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, id_):
        return dataset.ys[id_]
                
    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples

class Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, data, vocab, url_dic=None):
        'Initialization'
        self.unk_count = 0
        self.non_unk_count = 0
        self.ids, self.Xs, ys = data
        self.is_hyper2label = {"false":0, "true":1}
        if(ys is None):
            self.ys = None
        else:
            self.ys={}
            for k in ys:
                self.ys[k]=self.is_hyper2label[ys[k]]
#             self.ys = np.array(list(map(lambda label: self.is_hyper2label[label], self.ys.values())))

        self.vocab = vocab
        self.num_total_seqs = len(self.Xs)
        self.tt = MyTokenizer()
        self.emo = EmoFeatures(self.vocab, self.tt)
        # self.inputReducer = InputReducer()
        self.url_dic = url_dic

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.ids)

    def vectorize(self, sentence):
        idx_sequence = []
        for word in word_tokenize(sentence):
            if word in self.vocab.word2index:
                self.non_unk_count+=1
                idx_sequence.append(self.vocab.word2index[word]) 
            else:
                self.unk_count+=1
                idx_sequence.append(constant.UNK_idx)
        return idx_sequence

    def preprocess(self, sentnece):
        """Converts words to ids."""
        sent_ids = self.vectorize(sentnece.lower())
        return torch.LongTensor(sent_ids)
    
    def get_emo2vec(self, sentences):
        return self.emo.embedding(sentences)

    def __getitem__(self, id_ex):
        'Generates one sample of data'
        # Select sample
        id_ = self.ids[id_ex]
        # Load data and get label
        # X = self.preprocess(self.Xs[id_]['article_text'])
        ############ TODO online need to clean text first ############
        article_text = clean_txt(self.Xs[id_]['article_text'], remove_stopwords=False)
        # if constant.reduce_input:
        #     article_text = self.inputReducer.keyword_based(article_text)
        # title doesn't need to be cleaned
        title = self.Xs[id_]['title']
        if constant.use_bert:
            tit = title
            X = article_text
        else:
            tit = self.preprocess(title)
            X = self.preprocess(article_text)

        if(self.ys is None): y = None
        else: y = self.ys[id_]
        
        if constant.use_emo2vec_feat: 
            emo = self.get_emo2vec(title)
            # emo = self.get_emo2vec(article_text)         
            return X, tit, emo, y, id_

        if constant.use_url:
            url_1hot = self.url_dic.urls_to_idx_1_hot(self.Xs[id_]['external'])
            return X, tit, url_1hot, y, id_
        if constant.use_url_rnn:
            url = self.Xs[id_]['external'][0] if len(self.Xs[id_]['external'])!=0 else ' '
            domain = extract_base_url(url)
            domain_tensor = lineToChar1hot(domain)
#             print(domain_tensor)
            return X, tit, domain_tensor, y, id_
            
        return X, tit, y, id_
    
def collate_fn(data):
    def merge(sequences, max_length=300):
        lengths = [max_length if len(seq)>max_length else len(seq) for seq in sequences]

        padded_seqs = torch.ones(len(sequences), max(lengths)).long()
        for i, seq in enumerate(sequences):
#             seq = torch.tensor(seq)
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs, lengths 
        
    if not constant.use_bert:
        data.sort(key=lambda x: len(x[0]), reverse=True)
    if constant.use_emo2vec_feat or constant.use_url:
        X, tit, additional_feat, y, id_ = zip(*data)
        additional_feat = torch.Tensor(additional_feat)
    elif constant.use_url_rnn:
        X, tit, additional_feat, y, id_ = zip(*data)
#         print(additional_feat)
#         additional_feat = torch.Tensor(additional_feat)
    else:
        X, tit, y, id_ = zip(*data)

    if not constant.use_bert:
        tit, tit_len = merge(tit)
        X, x_len = merge(X)
        X = torch.LongTensor(X)
    else:
        tit_len = None
        x_len = None

    if(y[0] is None): pass
    else: y = torch.FloatTensor(y) #torch.LongTensor(y)
    if not constant.use_bert and constant.USE_CUDA:
        tit = tit.cuda()
        X = X.cuda()
        # something wrong with domain guess i did something wrong when i fixed the conflict.
        # domain = domain.cuda()
        if(y[0] is None): pass
        else: y = y.cuda()
        if constant.use_emo2vec_feat or constant.use_url: 
            additional_feat = additional_feat.cuda()
        if constant.use_url_rnn:
            # return X, x_len, tit, tit_len, domain, y, id_
            return X, x_len, tit, tit_len, None, y, id_
    return X, x_len, tit, tit_len, y, id_

def collate_fn_use_prob_or_publisher(data):
    # for training set, it needs to 
    def merge(sequences, max_length=500):
        lengths = [max_length if len(seq)>max_length else len(seq) for seq in sequences]
        padded_seqs = torch.ones(len(sequences), max(lengths)).long()
        for i, seq in enumerate(sequences):
#             seq = torch.tensor(seq)
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs, lengths 

    data.sort(key=lambda x: len(x[0]), reverse=True)
    X, tit, y, id_ = zip(*data)
    X, x_len = merge(X)
    tit, tit_len = merge(tit)
    X = torch.LongTensor(X)
    
    if constant.use_publisher == True:
        id_pub_label = constant.tr_pub_label
        feat = torch.Tensor(np.array([id_pub_label[item] for item in id_]))
    elif constant.use_topic_prob == True:
        id_topicProb = constant.id_topicProb
        feat = torch.Tensor(np.array([id_topicProb[item] for item in id_]))
    
    if(y[0] is None): pass
    else: y = torch.LongTensor(y)
    if constant.USE_CUDA:
        X = X.cuda() #X.cuda()
        feat = feat.cuda()
        if(y[0] is None): pass
        else: 
            y = y.cuda() #y.cuda()

    return X, x_len, tit, tit_len, y, feat, id_

def collate_fn_use_topicFeat_tr(data):
    # for training set, it needs to 
    def merge(sequences, max_length=500):
        lengths = [max_length if len(seq)>max_length else len(seq) for seq in sequences]

        padded_seqs = torch.ones(len(sequences), max(lengths)).long()
        for i, seq in enumerate(sequences):
#             seq = torch.tensor(seq)
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs, lengths 
    
    data.sort(key=lambda x: len(x[0]), reverse=True)
    X, tit, y, id_ = zip(*data)
    X, x_len = merge(X)
    tit, tit_len = merge(tit)
    X = torch.LongTensor(X)
    
    id_topicProb = constant.id_topicProb
    topic_prob = torch.Tensor(np.array([id_topicProb[item] for item in id_]))
    
    id_topicFeat = constant.id_topicFeat_train
    topic_feat = torch.Tensor(np.array([id_topicFeat[item] for item in id_]))

    if(y[0] is None): pass
    else: y = torch.LongTensor(y)
    if constant.USE_CUDA:
        tit = tit.cuda()
        X = X.cuda()
        topic_prob = topic_prob.cuda()
        topic_feat = topic_feat.cuda()
        if(y[0] is None): pass
        else: 
            y = y.cuda() #y.cuda()

    return X, x_len, tit, tit_len, y, topic_prob, topic_feat, id_

def collate_fn_use_topicFeat_test(data):
    # for training set, it needs to 
    def merge(sequences, max_length=500):
        lengths = [max_length if len(seq)>max_length else len(seq) for seq in sequences]

        padded_seqs = torch.ones(len(sequences), max(lengths)).long()
        for i, seq in enumerate(sequences):
#             seq = torch.tensor(seq)
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs, lengths 
    
    data.sort(key=lambda x: len(x[0]), reverse=True)
    X, tit, y, id_ = zip(*data)
    X, x_len = merge(X)
    tit, tit_len = merge(tit)
    X = torch.LongTensor(X)

    id_topicFeat = constant.id_topicFeat_test
    topic_feat = torch.Tensor(np.array([id_topicFeat[item] for item in id_]))

    if(y[0] is None): pass
    else: y = torch.LongTensor(y)
    if constant.USE_CUDA:
        tit = tit.cuda()
        X = X.cuda() #X.cuda()
        topic_feat = topic_feat.cuda()
        if(y[0] is None): pass
        else: 
            y = y.cuda() #y.cuda()

    return X, x_len, tit, tit_len, y, "", topic_feat, id_

def prepare_vocab():
    vocab_path = "/home/nayeon/fakenews/data_new/vocab_trim4.pickle"
    if Path(vocab_path).is_file() and not constant.build_vocab_flag:
        with open(vocab_path, 'rb') as handle:
            print("loading vocab from {}".format(vocab_path))
            vocab = pickle.load(handle)
            print("the length of vocab:", len(vocab.word2index))
#             # code for trimming.
#             trim_count=3
#             vocab.trim(min_frequency=trim_count)
#             with open('/home/nayeon/fakenews/data_new/vocab_trim{}.pickle'.format(trim_count), 'wb') as handle:
#                 pickle.dump(vocab, handle, pickle.HIGHEST_PROTOCOL)
#             print("Vocab size after trimming:",vocab.n_words)
    else:
        print("Generating Vocab first!!!")
        vocab = Lang()
        for xs_id in tqdm(xs_ids):
            vocab.index_words(xs[xs_id]['article_text'])
        print("Vocab size no dev:",vocab.n_words)
        for test_id in tqdm(test_ids):
            vocab.index_words(test_x[test_id]['article_text'])
        print("Vocab size with dev:",vocab.n_words)
        # save vocab
        if not constant.debug_mode:
            with open('./data_new/vocab.pickle', 'wb') as handle:
                pickle.dump(vocab, handle, pickle.HIGHEST_PROTOCOL)
    return vocab

def prepare_data(x_path, batch_size=32, is_shuffle=False):
    print("loading data from {}".format(x_path))
    train, val, test, url_dic = read_data(x_path, constant.debug_mode)

    vocab = prepare_vocab()
    
    dataset_train = Dataset(train, vocab, url_dic)
    # adversarial and multitask training needs topic probability
    if constant.use_topic_prob == True or constant.use_publisher == True:
        collate_fn_train = collate_fn_use_prob_or_publisher
        collate_fn_val = collate_fn
        collate_fn_test = collate_fn
    elif constant.use_topic_feat == True:
        collate_fn_train = collate_fn_use_topicFeat_tr
        collate_fn_val = collate_fn_use_topicFeat_tr
        collate_fn_test = collate_fn_use_topicFeat_test
    else:
        collate_fn_train = collate_fn
        collate_fn_val = collate_fn
        collate_fn_test = collate_fn

    data_loader_tr = torch.utils.data.DataLoader(dataset=dataset_train,
                                                batch_size=batch_size, 
                                                collate_fn=collate_fn_train,
                                                shuffle=True)
#                                                     sampler = ImbalancedDatasetSampler(dataset_train))

    dataset_val = Dataset(val, vocab, url_dic)
    data_loader_val = torch.utils.data.DataLoader(dataset=dataset_val,
                                                batch_size=batch_size if not constant.use_bert else 4,
                                                shuffle=False, 
                                                collate_fn=collate_fn_val)

    dataset_test = Dataset(test, vocab, url_dic)
    data_loader_test = torch.utils.data.DataLoader(dataset=dataset_test,
                                                batch_size=batch_size if not constant.use_bert else 4,
                                                shuffle=False, 
                                                collate_fn=collate_fn_test)

    return data_loader_tr, data_loader_val, data_loader_test, vocab

def prepare_filtered_data(batch_size=32):
    print("loading filtered train and filtered test ...")
    
    filtered_train, filtered_test = read_filtered_data()
    vocab = prepare_vocab()
    
    dataset_train = Dataset(filtered_train, vocab)
    data_loader_train = torch.utils.data.DataLoader(dataset=dataset_train,
                                                batch_size=batch_size,
                                                shuffle=False, 
                                                collate_fn=collate_fn,
                                                sampler = ImbalancedDatasetSampler(dataset_train))
    
    dataset_test = Dataset(filtered_test, vocab)
    data_loader_test = torch.utils.data.DataLoader(dataset=dataset_test,
                                                batch_size=batch_size,
                                                shuffle=False, 
                                                collate_fn=collate_fn)
    
    return data_loader_train, data_loader_test, vocab

def prepare_byarticle_data(aug_count="", batch_size=32, test_whole=False):
    print("loading final test data ...")
    vocab = prepare_vocab()

    if aug_count!="":
        train, test = read_byarticle_aug_data(aug_count)
    else:
        if test_whole == True:
            test = read_byarticle_data(test_whole)
            dataset_test = Dataset(test, vocab)
            data_loader_test = torch.utils.data.DataLoader(dataset=dataset_test,
                                                        batch_size=batch_size,
                                                        shuffle=False, 
                                                        collate_fn=collate_fn)
            return data_loader_test
        else:
            train, test = read_byarticle_data(test_whole)
            dataset_train = Dataset(train, vocab)
            data_loader_train = torch.utils.data.DataLoader(dataset=dataset_train,
                                                        batch_size=batch_size,
                                                        shuffle=False, 
                                                        collate_fn=collate_fn,
                                                        sampler = ImbalancedDatasetSampler(dataset_train))
            
            dataset_test = Dataset(test, vocab)
            data_loader_test = torch.utils.data.DataLoader(dataset=dataset_test,
                                                        batch_size=batch_size,
                                                        shuffle=False, 
                                                        collate_fn=collate_fn)
            
            return data_loader_train, data_loader_test, vocab

def prepare_byarticle_kfold(ids_train, data_train, labels_train, batch_size=32):
    ids_file = open("data_new/by_article_ids_test.pickle", "rb")
    ids_test = pickle.load(ids_file)
    data_file = open("data_new/preprocessed_byarticle_test.pickle", "rb")
    data_test = pickle.load(data_file)
    labels_file = open("data_new/by_article_labels_test.pickle", "rb")
    labels_test = pickle.load(labels_file)
    
    vocab_path = "/home/nayeon/fakenews/data_new/vocab_trim4.pickle"
    if Path(vocab_path).is_file() and not constant.build_vocab_flag:
        with open(vocab_path, 'rb') as handle:
            print("loading vocab from {}".format(vocab_path))
            vocab = pickle.load(handle)
            print("the length of vocab:", len(vocab.word2index))
    else:
        print("getting a vocab first !!")
    
    train = (ids_train, data_train, labels_train)
    test = (ids_test, data_test, labels_test)
    
    dataset_train = Dataset(train, vocab)
    data_loader_train = torch.utils.data.DataLoader(dataset=dataset_train,
                                                batch_size=batch_size if not constant.use_bert else 4,
                                                shuffle=False, 
                                                collate_fn=collate_fn,
                                                sampler = ImbalancedDatasetSampler(dataset_train))
    
    dataset_test = Dataset(test, vocab)
    data_loader_test = torch.utils.data.DataLoader(dataset=dataset_test,
                                                batch_size=batch_size if not constant.use_bert else 4,
                                                shuffle=False, 
                                                collate_fn=collate_fn)
    
    return data_loader_train, data_loader_test, vocab

def prepare_byarticle_cross_validation(train, val, test, batch_size=32, aug_count=''):
    
    vocab_path = "/home/nayeon/fakenews/data_new/vocab_trim4.pickle"
    if Path(vocab_path).is_file() and not constant.build_vocab_flag:
        with open(vocab_path, 'rb') as handle:
            print("loading vocab from {}".format(vocab_path))
            vocab = pickle.load(handle)
            print("the length of vocab:", len(vocab.word2index))
    else:
        print("getting a vocab first !!")
    
    if aug_count != '':
        train, val, test, ids_val_dict, ids_test_dict = read_byarticle_aug_data_for_cross_validation(train, val, test, aug_count)

    dataset_train = Dataset(train, vocab)
    data_loader_train = torch.utils.data.DataLoader(dataset=dataset_train,
                                                batch_size=batch_size if not constant.use_bert else 16,
                                                shuffle=False, 
                                                collate_fn=collate_fn,
                                                sampler = ImbalancedDatasetSampler(dataset_train))

    dataset_val = Dataset(val, vocab)
    data_loader_val = torch.utils.data.DataLoader(dataset=dataset_val,
                                                batch_size=batch_size if not constant.use_bert else 16,
                                                shuffle=False, 
                                                collate_fn=collate_fn)
    
    dataset_test = Dataset(test, vocab)
    data_loader_test = torch.utils.data.DataLoader(dataset=dataset_test,
                                                batch_size=batch_size if not constant.use_bert else 16,
                                                shuffle=False, 
                                                collate_fn=collate_fn)

    if aug_count != '':
        return data_loader_train, data_loader_val, data_loader_test, ids_val_dict, ids_test_dict
    else:
        return data_loader_train, data_loader_val, data_loader_test

def prepare_nfolds_cleaner_data(train, test, batch_size=16):
    vocab_path = "/home/nayeon/fakenews/data_new/vocab_trim4.pickle"
    if Path(vocab_path).is_file() and not constant.build_vocab_flag:
        with open(vocab_path, 'rb') as handle:
            print("loading vocab from {}".format(vocab_path))
            vocab = pickle.load(handle)
            print("the length of vocab:", len(vocab.word2index))
    else:
        print("getting a vocab first !!")
    
    dataset_train = Dataset(train, vocab)
    data_loader_train = torch.utils.data.DataLoader(dataset=dataset_train,
                                                batch_size=batch_size,
                                                shuffle=False, 
                                                collate_fn=collate_fn,
                                                sampler = ImbalancedDatasetSampler(dataset_train))
    
    dataset_test = Dataset(test, vocab)
    data_loader_test = torch.utils.data.DataLoader(dataset=dataset_test,
                                                batch_size=batch_size,
                                                shuffle=False, 
                                                collate_fn=collate_fn)

    return data_loader_train, data_loader_test