#!/bin/python
#authour: alphaplato
#describe: seq2seq

from gensim.models import Word2Vec
from gensim.corpora.dictionary import Dictionary
from keras.preprocessing import sequence
from keras.models import Model
from keras.layers.embeddings import Embedding
from keras.layers import Input,LSTM,Dense,Dot,Reshape,TimeDistributed
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

import nltk
import multiprocessing
import os
import pandas as pd
import numpy as np

import tensorflow as tf
#from tensorflow.keras  import backend as K  #  auc使用
import keras.backend.tensorflow_backend as K
#K.set_session(tf.Session(config=tf.ConfigProto(device_count={'gpu':1})))
import logging

cup_count=multiprocessing.cpu_count()

vocab_dim=128
window_size=4
min_count=1
n_iteration=4
sg=1

max_len=50
batch_size=64
n_epoch=4

h_dim=128

log_level=logging.INFO  #DEBUG可以输出详细信
LOG_FORMAT="%(asctime)s-%(levelname)s-%(message)s"
DATE_FORMAT="%m/%d/%Y  %H:%M:%S"
logging.basicConfig(level=logging.INFO,format=LOG_FORMAT)
logger=logging.getLogger()

def  load_data(data):
        logger.info("loading and tokenize...")
        dataset=pd.read_csv(data,header=0,names=['en','ch','source'],sep='\t').dropna()
        sent_data=dataset[['ch','en']].applymap(lambda x:'<bg>\t'+x)
        sent_data=sent_data[['ch','en']].applymap(nltk.word_tokenize)
        target_data=dataset[['en']].applymap(lambda x: x+'\t<eos>')
        target_data=target_data[['en']].applymap(nltk.word_tokenize)
        return  sent_data,target_data  

def  get_params(model):
        gensim_dict=Dictionary()
        gensim_dict.doc2bow(model.wv.vocab.keys(),
                    allow_update=True)
        w2index={w:k+1 for k,w in gensim_dict.items()}
        n_symbols=len(w2index)+1
        embed_weights=np.zeros((n_symbols,vocab_dim))
        for  w,k  in  w2index.items():
                embed_weights[k,:]=model[w]
        logger.info("the total words,n_symbols:{0}".format(n_symbols))
        return embed_weights,w2index,n_symbols

def input_data(w2index,sent_data):
    def func(sent):
        sent_sequence=[]
        for w in sent:
            if w in w2index:
                sent_sequence.append(w2index[w])
            else:
                sent_sequence.append(0)
        return sent_sequence
    sent_data=sent_data.applymap(func)
    return sent_data

def input_target(target_data):
    en2set=set()
    for sent in target_data.en.tolist():
        en2set=en2set | set(sent)
    n_en_sympols=len(en2set)
    en2index=dict(zip(en2set,range(n_en_sympols)))
    logger.info("the total english words,n_en_sympols:{0}".format(n_en_sympols))

    def func(sent):
        target_sequence=[]
        for w in sent:
            one_hot=[0]*n_en_sympols
            if w in en2index:
                target_sequence.append(en2index[w])
            else:
                target_sequence.append(0)
        return target_sequence
    target_data=target_data.applymap(func)
    return target_data,n_en_sympols

def split_data(data, target):
    target=sequence.pad_sequences(target.en.tolist(),maxlen=max_len)
    target=np.expand_dims(target,-1)
    logger.info("the shape target:{0}".format(target.shape))
    X_train,X_test,Y_train,Y_test=train_test_split(data,target,test_size=0.1, random_state=0) 
    ch_input_train=sequence.pad_sequences(X_train.ch.tolist(),maxlen=max_len)
    en_input_train=sequence.pad_sequences(X_train.en.tolist(),maxlen=max_len)
    ch_input_test=sequence.pad_sequences(X_test.ch.tolist(),maxlen=max_len)
    en_input_test=sequence.pad_sequences(X_test.en.tolist(),maxlen=max_len)
    target_train=Y_train
    target_test=Y_test
    return ch_input_train,en_input_train,ch_input_test,en_input_test,target_train,target_test

def word2vec(common_texts):
    model=Word2Vec(size=vocab_dim, window=window_size, min_count=min_count, workers=cup_count, iter=n_iteration)
    model.build_vocab(common_texts)
    logger.info("word2vec...")
    logger.info("model.corpus_count:{0},model.epochs:{1}".format(model.corpus_count,model.epochs))
    model.train(common_texts,total_examples=model.corpus_count, epochs=model.epochs)
    model.save('model/word2vec.pkl')
    embed_weights,w2index,n_symbols=get_params(model)
    return embed_weights,w2index,n_symbols

def get_model(n_symbols,n_en_sympols,embed_weights):
    embed=Embedding(output_dim=vocab_dim,
        input_dim=n_symbols,
        weights=[embed_weights],
        input_length=max_len)
    #encode 
    ch_input=Input(shape=(None,),dtype='int32')  # input ch sequence
    ch_sent=embed(ch_input)
    encoder_outputs,state_h,state_c=LSTM(units=h_dim,return_state=True)(ch_sent)
    encoder_states=[state_h,state_c]
    #decode
    en_input=Input(shape=(None,),dtype='int32') # input en sequence
    en_sent=embed(en_input)
    decoder_outputs=LSTM(units=h_dim,return_sequences=True)(en_sent,initial_state=encoder_states)
    decoder_outputs=TimeDistributed(Dense(n_en_sympols,activation='softmax'))(decoder_outputs)
    #decoder_outputs=Dense(units=n_en_sympols, activation='softmax')(decoder_outputs)
    model=Model(inputs=[ch_input,en_input],outputs=decoder_outputs)
    logger.info("model summary:{0}".format(model.summary()))
    return ch_input,en_input,model

def seq2seq_model(sent_data,target_data,n_sympols,w2index,embed_weights):
    sent_data=input_data(w2index,sent_data)
    target_data,n_en_sympols=input_target(target_data)
    ch_input_train,en_input_train,ch_input_test,en_input_test,target_train,target_test=split_data(sent_data,target_data)

    ch_input,en_input,model=get_model(n_sympols,n_en_sympols,embed_weights)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])
    logging.info("Seq2seq train...")
    model.fit([ch_input_train,en_input_train],target_train,batch_size=batch_size,epochs=n_epoch,verbose=1)
    logging.info("Seq2seq save...")
    model.save('model/seq2seq_model.h5')
    logging.info("Seq2seq evaluate...")
    evaluations=model.evaluate([ch_input_test,en_input_test],target_test, batch_size=batch_size)
    logger.info("final evaluations:{0}".format(evaluations))

logger.setLevel(log_level)
sent_data,target_data=load_data('data/cmn.txt') 
common_texts=sent_data.ch.tolist() + target_data.en.tolist()
embed_weights,w2index,n_symbols=word2vec(common_texts)
seq2seq_model(sent_data,target_data,n_symbols,w2index,embed_weights)