#!/bin/python
#authour: alphaplato
#describe: dssm model

from gensim.models import Word2Vec
from gensim.corpora.dictionary import Dictionary
from keras.preprocessing import sequence
from keras.models import Model
from keras.layers.embeddings import Embedding
from keras.layers import Input,Conv1D,MaxPooling1D,Dense,Dot,Reshape,Flatten
from sklearn.model_selection import train_test_split

import nltk
import multiprocessing
import os
import pandas as pd
import numpy as np

import tensorflow as tf
#from tensorflow.keras import backend as K # auc使用
import keras.backend.tensorflow_backend as K
#K.set_session(tf.Session(config=tf.ConfigProto(device_count={'gpu':1})))
import logging

cup_count=multiprocessing.cpu_count()

vocab_dim = 128
window_size = 4
min_count = 5
n_iteration = 1

max_len =100
batch_size = 64
n_epoch = 4

log_level = logging.INFO #DEBUG可以输出详细信
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger=logging.getLogger()

def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc

def load_data(data):
    dataset = pd.read_csv(data).dropna()
    logger.info("loading and tokenize...")
    questionA = list(map(nltk.word_tokenize,dataset.question1.tolist()))
    questionB = list(map(nltk.word_tokenize,dataset.question2.tolist()))
    labels = list(map(int,dataset.is_duplicate.tolist()))
    return questionA,questionB,labels

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

def input_data(w2index,questionA,questionB):
    def func(sent):
        sent_sequence = []
        for w in sent:
            if w in w2index:
                sent_sequence.append(w2index[w])
            else:
                sent_sequence.append(0)
        return sent_sequence
    questionA = sequence.pad_sequences(list(map(func, questionA)),maxlen=max_len)
    questionB = sequence.pad_sequences(list(map(func, questionB)),maxlen=max_len)
    return questionA,questionB
        

def word2vec(common_texts):
    model = Word2Vec(size=vocab_dim, window=window_size, min_count=min_count, workers=cup_count, iter=n_iteration)
    model.build_vocab(common_texts)
    logger.info("word2vec...")
    logger.info("model.corpus_count:{0},model.epochs:{1}".format(model.corpus_count,model.epochs))
    model.train(common_texts,total_examples=model.corpus_count, epochs=model.epochs)
    model.save('model/word2vec.pkl')
    embed_weights,w2index,n_symbols = get_params(model)
    return embed_weights,w2index,n_symbols


def get_model(n_symbols,embed_weights):  
    my_input = Input(shape=(max_len,),dtype='int32') # input sequence
    sent = Embedding(output_dim=vocab_dim,
        input_dim=n_symbols,
        weights=[embed_weights],
        input_length=max_len)(my_input)
    convontion = Conv1D(256,3,activation='relu',padding='same')(sent)
    maxpooling = MaxPooling1D(max_len,padding='same')(convontion)
    dense = Dense(units=128, activation='relu')(maxpooling)
    output = Flatten()(dense)
    model = Model(my_input,output)
    return model

def dssm(questionA,questionB,labels,n_symbols,embed_weights):
    model = get_model(n_symbols,embed_weights)
    inputA = Input(shape=(max_len,),dtype='int32')
    inputB = Input(shape=(max_len,),dtype='int32')
    outputA = model(inputs=inputA)
    outputB = model(inputs=inputB)
    output = Dot(axes=1,normalize=True)([outputA,outputB])
    output = Dense(units=1,activation='hard_sigmod')
    dssm_model = Model(inputs=[inputA,inputB],output=output)
    dssm_model.compile(optimizer='rmsprop', loss='binary_crossentropy',
              metrics=[auc])
    logging.info("DSSM train...")
    dssm_model.fit([questionA,questionB],labels, batch_size=batch_size, epochs=n_epoch,verbose=1)
    logging.info("DSSM save...")
    dssm_model.save('model/dssm_model.h5')

logger.setLevel(log_level)
questionA,questionB,labels = load_data('data/train.csv_1') 
embed_weights,w2index,n_symbols = word2vec(questionA+questionB)
questionA,questionB = input_data(w2index,questionA,questionB)
dssm(questionA,questionB,labels,n_symbols,embed_weights)