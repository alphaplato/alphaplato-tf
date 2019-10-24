#!/bin/python
#authour: alphaplato
#describe: dssm model

from gensim.models import Word2Vec
from gensim.corpora.dictionary import Dictionary
from keras.preprocessing import sequence
from keras.models import Model
from keras.layers.embeddings import Embedding
from keras.layers import Input,LSTM,Dense,Dot,Flatten
from sklearn.model_selection import train_test_split

import nltk
import multiprocessing
import os
import pandas as pd
import numpy as np

import tensorflow as tf
from keras import backend as K # auc使用
#K.set_session(tf.Session(config=tf.ConfigProto(device_count={'gpu':1})))
import logging

cup_count=multiprocessing.cpu_count()

vocab_dim = 128
window_size = 4
min_count = 5
n_iteration = 4

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
    logger.info("loading and tokenize...")
    dataset = pd.read_csv(data).dropna()
    data = dataset[['question1','question2']].applymap(nltk.word_tokenize)
    label = dataset['is_duplicate'].apply(int).tolist()
    X_train,X_test,Y_train,Y_test = train_test_split(data,label,test_size=0.1, random_state=0)
    return X_train,X_test,Y_train,Y_test

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

def input_data(w2index,dataset):
    def func(sent):
        sent_sequence = []
        for w in sent:
            if w in w2index:
                sent_sequence.append(w2index[w])
            else:
                sent_sequence.append(0)
        return sent_sequence
    questionA = sequence.pad_sequences(list(map(func, dataset.question1.tolist())),maxlen=max_len)
    questionB = sequence.pad_sequences(list(map(func, dataset.question2)),maxlen=max_len)
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
    my_input = Input(shape=(max_len,),dtype='int32')  # input sequence
    sent = Embedding(output_dim=vocab_dim,
        input_dim=n_symbols,
        weights=[embed_weights],
        input_length=max_len)(my_input)
    lstm = LSTM(units=256, return_sequences=True)(sent)
    output = LSTM(units=128)(lstm)
    return my_input,output

def dssm_model(X_train,X_test,Y_train,Y_test,n_symbols,w2index,embed_weights):
    my_inputA,outputA = get_model(n_symbols,embed_weights)
    my_inputB,outputB = get_model(n_symbols,embed_weights)
    output = Dot(axes = 1,normalize=True)([outputA,outputB])
    output = Dense(units=1,activation='hard_sigmoid')(output)
    dssm_model = Model(inputs=[my_inputA,my_inputB],output=output)
    dssm_model.compile(optimizer='rmsprop', loss='binary_crossentropy',
              metrics=[auc])
    logging.info("DSSM train...")
    questionA,questionB = input_data(w2index,X_train)
    labels = Y_train
    dssm_model.fit([questionA,questionB],labels, batch_size=batch_size, epochs=n_epoch,verbose=1)
    logging.info("DSSM save...")
    dssm_model.save('model/dssm_model.h5')

    logging.info("DSSM evaluate...")
    questionA,questionB = input_data(w2index,X_test)
    labels = Y_test 
    dssm_model.fit([questionA,questionB],labels, batch_size=batch_size)

logger.setLevel(log_level)
X_train,X_test,Y_train,Y_test = load_data('data/train.csv') 
common_texts = X_train.question1.tolist() + X_train.question2.tolist()
embed_weights,w2index,n_symbols = word2vec(common_texts)
dssm_model(X_train,X_test,Y_train,Y_test,n_symbols,w2index,embed_weights)