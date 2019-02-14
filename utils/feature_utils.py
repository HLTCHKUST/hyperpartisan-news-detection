import torch
import torch.nn as nn
import numpy as np
from gensim.models.keyedvectors import KeyedVectors
import os
from urllib.parse import urlparse

def create_embedding_matrix(original_vocab, path, embedding_dim,
                            embedding_type="word2vec", use_torch=True):
    print("loading %s embeddings from file %s" % (embedding_type, path))
    cnt_total = 0
    cnt_pretrained = 0
    if embedding_type == 'word2vec':
        wv = KeyedVectors.load_word2vec_format(path, binary=True)
        # print (wv.vocab)
        embedding_matrix = np.zeros((len(original_vocab), int(embedding_dim)))
        for word in list(original_vocab.word2index.keys()):
            cnt_total += 1
            if word in wv.vocab:
                # original_vocab.get(word) = get id of the vocab
                cnt_pretrained += 1
                embedding_matrix[original_vocab.word2index[word], :] = wv[word]

        assert embedding_matrix.shape[0] == len(original_vocab)
        assert embedding_matrix.shape[1] == embedding_dim
        print('Pre-trained: %d (%.2f%%)' % (cnt_pretrained, cnt_pretrained * 100.0 / original_vocab.n_words))
        
    elif embedding_type == "fasttext":
        # load from file
        lines = [line.rstrip() for line in open(path, "r")]
        vocab_size = len(original_vocab)
        embedding_matrix = np.zeros((vocab_size, int(embedding_dim)))

        overlap_count = 0
        for i, l in enumerate(lines):
            if i == 0:
                vocab_size, dim = l.split()
                assert int(dim) == embedding_dim
            else:
                values = l.split()
                assert len(values) == int(dim) + 1
                index = original_vocab.get(values[0])
                if index > 0:
                    embedding_matrix[index] = [float(v) for v in values[1:]]
                    overlap_count += 1
            
        print("added pretrained embedding for %s words" % overlap_count)
    else:
        raise ValueError("Wrong word vector name")

    print("finished loading. embedding matrix shape=%s" %
          str(embedding_matrix.shape))

    if use_torch:
        return torch.from_numpy(embedding_matrix).type(torch.FloatTensor)
    return embedding_matrix


def gen_embeddings(vocab, embedding_dim, emb_file):
    """
        Generate an initial embedding matrix for `word_dict`.
        If an embedding file is not given or a word is not in the embedding file,
        a randomly initialized vector will be used.
    """
    embeddings = np.zeros((vocab.n_words, embedding_dim))
    print('Embeddings: %d x %d' % (vocab.n_words, embedding_dim))
    if emb_file is not None:
        print('Loading embedding file: %s' % emb_file)
        pre_trained = 0
        for line in open(emb_file).readlines():  # each iter gets one word and its embedding
            sp = line.split()
            if(len(sp) == embedding_dim + 1):
                if sp[0] in vocab.word2index:
                    pre_trained += 1
                    embeddings[vocab.word2index[sp[0]]] = [float(x) for x in sp[1:]]
            else:
                print("Error:",sp[0])
        print('Pre-trained: %d (%.2f%%)' % (pre_trained, pre_trained * 100.0 / vocab.n_words))
    return embeddings

def get_emo_embedding(lang):
    if not os.path.exists("/home/nayeon/fakenews/vectors/emo2vec/vocab.pkl") or not os.path.exists("/home/nayeon/fakenews/vectors/emo2vec/emo2vec.pkl"):
        from utils.download_google_drive import download_file_from_google_drive
        file_id = '1K0RPGSlBHOng4NN4Jkju_OkYtrmqimLi'
        destination = 'vectors/emo2vec.zip'
        download_file_from_google_drive(file_id, destination)
        print("start unzipping")
        import zipfile
        zip_ref = zipfile.ZipFile(destination, 'r')
        zip_ref.extractall("vectors/")
        zip_ref.close()
        os.remove(destination)
        
    import pickle
    emo_size = 100
    emo2vec_vocab = "/home/nayeon/fakenews/vectors/emo2vec/vocab.pkl"
    emo2vec_emb = "/home/nayeon/fakenews/vectors/emo2vec/emo2vec.pkl"
    with open(emo2vec_vocab, 'rb') as f:
        emo_word2id = pickle.load(f, encoding="latin1")["word2id"]

    with open(emo2vec_emb,'rb') as f:
        emo_embedding = pickle.load(f, encoding='latin1')

    # new_embedding = np.zeros((100, emb_size))
    new_embedding = np.zeros((lang.n_words, emo_size))
    for i in range(lang.n_words):
    # for i in range(100):
        if emo_size > 0 and lang.index2word[i] in emo_word2id:
            new_embedding[i] = emo_embedding[emo_word2id[lang.index2word[i]]]

    return new_embedding


class EmoFeatures(object):
    def __init__(self, vocab, tokenizer):
        self.tokenizer = tokenizer
        self.emo_emb = get_emo_embedding(vocab)
        self.word2index = vocab.word2index
        self.index2word = vocab.index2word

    def embedding(self, sent, mode="full"):
        emb = []
        if sent=="":
            emb.append(np.zeros((100,)))
        else:
            for w in self.tokenizer.tokenize(sent):
                if w in self.word2index:
                    emb.append(self.emo_emb[self.word2index[w]])
                else:
                    emb.append(np.zeros((100,)))
        if mode == "sum":
            return np.sum(emb, axis=0)
        elif mode == "avg" or "average":
            return np.mean(emb, axis=0)
        elif mode == "max":
            return np.max(np.array(emb), axis=0)
        else:
            raise ValueError("invalid mode arguments")

def get_url_feature_utils(url):
    data = urlparse(url)
    return data.netloc

