#! /bin/env python
# -*- coding: utf-8 -*-
#authour: alphaplato
#describe: dssm model

import pandas as pd 
import numpy as np 
import multiprocessing
import nltk
import keras

from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary
from keras.preprocessing import sequence

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout,Activation
from keras.models import model_from_yaml
np.random.seed(1337)  # For Reproducibility
import sys
sys.setrecursionlimit(1000000)
import yaml
import logging

import tensorflow as tf
from keras import backend as K

log_level = logging.INFO #DEBUG可以输出详细信
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger=logging.getLogger()

# set parameters:
cpu_count = multiprocessing.cpu_count() # 4
vocab_dim = 32
window_size = 4
min_count = 5
n_iteration = 4

max_len =20
batch_size = 64
n_epoch = 4

batch_size = 32

def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc

def load_data(data):
    logger.info('loading data and tokenize word...')
    dataset = pd.read_csv(data,encoding = 'ISO-8859-1')
    texts = tokenizer(dataset.SentimentText.tolist())
    labels = list(map(int,dataset.Sentiment.tolist()))
    return texts,labels

#对句子经行分词，并去掉换行符
def tokenizer(texts):
    return list(map(nltk.word_tokenize,texts))

def get_params(model):
    gensim_dict = Dictionary()
    gensim_dict.doc2bow(model.wv.vocab.keys(),
                            allow_update=True)
    w2index = {w: k+1 for k,w in gensim_dict.items()}
    n_symbols = len(w2index) + 1
    embed_weights = np.zeros((n_symbols,vocab_dim))
    for w,k in w2index.items():
        embed_weights[k,:] = model[w]
    print("the total words,n_symbols:{0}".format(n_symbols))
    return embed_weights,w2index,n_symbols

#创建词语字典，并返回每个词语的索引，词向量，以及每个句子所对应的词语索引
def word2vec(common_texts):
    model = Word2Vec(size=vocab_dim, window=window_size,
                        min_count=min_count,
                        workers=cpu_count, 
                        iter=n_iteration)
    model.build_vocab(common_texts)
    logger.info("word2vec...")
    logger.info("model.corpus_count:{0},model.epochs:{1}".format(model.corpus_count,model.epochs))
    model.train(common_texts,total_examples=model.corpus_count, epochs=model.epochs)
    model.save('model/word2vec.pkl')
    embed_weights,w2index,n_symbols = get_params(model)
    return embed_weights,w2index,n_symbols

def input_data(w2index,texts):
    def func(sent):
        sent_sequence = []
        for w in sent:
            if w in w2index:
                sent_sequence.append(w2index[w])
            else:
                sent_sequence.append(0)
        return sent_sequence
    texts = sequence.pad_sequences(list(map(func, texts)),maxlen=max_len)
    return texts

def get_model(n_symbols,embed_weights):
    logger.info('defining model...')
    model = Sequential()  # or Graph or whatever
    model.add(Embedding(output_dim=vocab_dim,
                        input_dim=n_symbols,
                        mask_zero=True,
                        weights=[embed_weights],
                        input_length=max_len))  # Adding Input Length
    model.add(LSTM(units=256, activation='tanh',return_sequences=True))
    model.add(LSTM(units=128, activation='tanh'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1,activation='hard_sigmoid'))
    return model

##定义网络结构
def train_lstm(n_symbols,embedding_weights,texts,labels):
    model = get_model(n_symbols,embedding_weights)
    texts = input_data(w2index,texts)
    x_train, x_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2)
    logger.info('compiling the Model...')
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',metrics=[auc])

    logger.info("Train...") # batch_size=32
    model.fit(x_train, y_train, batch_size=batch_size, epochs=n_epoch,verbose=1)

    logger.info("Evaluate...")
    score = model.evaluate(x_test, y_test,
                                batch_size=batch_size)
    logger.info('Test score:{0}'.format(score))
    #preds = model.predict(x_test, batch_size=batch_size)
    yaml_string = model.to_yaml()
    with open('model/lstm.yml', 'w') as outfile:
        outfile.write( yaml.dump(yaml_string, default_flow_style=True) )
    model.save_weights('model/lstm.h5')

logger.setLevel(log_level)
#训练模型，并保存
texts,labels=load_data('data/dataset.csv')
embed_weights,w2index,n_symbols = word2vec(texts)
train_lstm(n_symbols,embed_weights,texts,labels)