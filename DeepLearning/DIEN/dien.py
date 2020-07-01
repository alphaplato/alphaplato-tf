#!/bin/python3
#coding:utf-8
#Copyright 2020 Alphaplato. All Rights Reserved.
#Desc:dien model
#=======================================================

import tensorflow as tf
from rnn import dynamic_rnn
from rnn_cell_utils import AUGRUCell
import random

class DIEN(object):
    def __init__(self,fg):
        self._model_name = 'DIEN'
        self._fg = fg

    def _neg_sampling(self,SparseTensor,mid_cat,neg_count):
        def _sampling(values,
                      indices,
                      dense_shape,
                      counter=5): 
            mid_list=list(mid_cat.keys())    
            new_indices = []
            noclk_tmp_mid = []
            noclk_tmp_cat = []

            for indice in indices:
                if indice[1] == 0:
                    continue
                index = indice[0] + indice[1]
                mid = values[index]

                for i in range(counter):
                    index = random.randint(0, len(mid_list) - 1)
                    noclk_mid = mid_list[index]
            
                    if noclk_mid == mid:
                        continue
                    noclk_tmp_mid.append(noclk_mid)
                    noclk_tmp_cat.append(mid_cat[noclk_mid])
                    new_indices.append([indice[0],(indice[1]-1)*5+i])

            dense_shape = [dense_shape[0],(dense_shape[1]-1)*counter]

            return (noclk_tmp_mid,noclk_tmp_cat,new_indices,dense_shape)

        values = SparseTensor.values
        indices = SparseTensor.indices
        dense_shape = SparseTensor.dense_shape

        noclk_tmp_mid,noclk_tmp_cat,new_indices,dense_shape = tf.py_func(
                _sampling,
                [values,
                 indices,
                 dense_shape,
                 neg_count],
                Tout=[tf.string,tf.string,tf.int64,tf.int64])
        noclk_mid = tf.SparseTensorValue(indices=new_indices,values=noclk_tmp_mid,dense_shape=dense_shape)
        noclk_cat = tf.SparseTensorValue(indices=new_indices,values=noclk_tmp_cat,dense_shape=dense_shape)
        features_neg={'item_list':noclk_mid,'item_cat_list':noclk_cat}
        return features_neg
        

    def _gru_rnn(self,inputs,inputs_length):
        rnn_hidden_size = inputs.get_shape()[2]
        cell = tf.nn.rnn_cell.GRUCell(num_units=rnn_hidden_size)
        outputs, last_states = tf.nn.dynamic_rnn(
            cell=cell,
            dtype=tf.float32,
            sequence_length= inputs_length,
            inputs=inputs,
            time_major=False
        )
        return outputs, last_states

    def _augru_rnn(self,queries,inputs,inputs_length,params):
        rnn_hidden_size = inputs.get_shape()[2]
        att_scores = self._attention(queries,inputs,inputs_length,params)

        cell = AUGRUCell(num_units=rnn_hidden_size)
        outputs, last_states = dynamic_rnn(
            cell=cell,
            att_scores = att_scores,
            dtype=tf.float32,
            sequence_length= inputs_length,
            inputs=inputs,
            time_major=False
            )
        return outputs, last_states

    def _auxiliary_loss(self,inputs,params,item_inputs_len,is_pos=True):
        aux_layers = params['auxili_layers']
        l2_reg = params['l2_reg']
        with tf.variable_scope("aux_net",reuse=tf.AUTO_REUSE): 
            aux_input = inputs
            for layer in aux_layers:
                aux_input = tf.layers.dense(aux_input,
                                            layer,
                                            activation=tf.nn.relu,name='fl_aux_{0}'.format(layer),
                                            kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg))
            y_out = tf.layers.dense(aux_input,
                                    1,
                                    activation=tf.nn.sigmoid,
                                    name='f1_aux_out',
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg))
        y_out = tf.squeeze(y_out)
        if is_pos:
            y_loss = -tf.log(y_out)
        else:
            y_loss = -tf.log(1-y_out)
        pos_padding = tf.zeros_like(y_out)
        mask = tf.sequence_mask(item_inputs_len,tf.shape(aux_input)[1])
        mask_loss = tf.where(mask,y_loss,pos_padding)
        return mask_loss


    def _attention(self,queries,keys,keys_length,params):
        atten_layers = params['atten_layers']
        keys_shape = tf.shape(keys) #B*T*H
        queries = tf.tile(queries,[1,keys_shape[1]])
        queries = tf.reshape(queries,[keys_shape[0],keys_shape[1],keys.get_shape().as_list()[2]])
        # 将batch_size*keys_length当作batch处理，即在embedding层面做处理
        inputs = tf.concat([queries,keys,queries-keys],axis=-1)
        for layer in atten_layers:
            inputs = tf.layers.dense(inputs, layer, activation=tf.nn.relu, name='fl_att_{0}'.format(layer))
        att_scores = tf.layers.dense(inputs, 1, activation=tf.nn.sigmoid, name='f1_att_weight')
        
        att_scores = tf.squeeze(att_scores)
        key_masks = tf.sequence_mask(keys_length,keys_shape[1])
        paddings = tf.ones_like(att_scores) * (-2 ** 32 + 1) 
        att_scores = tf.where(key_masks,att_scores,paddings)
        att_scores = tf.nn.softmax(att_scores)
        att_scores = tf.expand_dims(att_scores,2)

        return att_scores
    
    def build_logits(self,features,labels,params,mode=tf.estimator.ModeKeys.TRAIN):
        feature_columns = self._fg.feature_columns
        layers = params['fcn_layers']
        dropout = params['dropout']
        l2_reg = params['l2_reg']
        neg_count = params['neg_count']
        mid_cat = params['mid_cat']

        with tf.variable_scope("dien-net"):
            with tf.variable_scope("attention-rnn"):
                user_id = tf.feature_column.input_layer(features,feature_columns['user_id'])
                item_input = tf.feature_column.input_layer(features,[feature_columns['item_id'],feature_columns['item_cat']])
                item_inputs,item_inputs_len = tf.contrib.feature_column.sequence_input_layer(features,[feature_columns['item_list'],feature_columns['item_cat_list']])
                gru_item_outputs, gru_item_last_states=self._gru_rnn(item_inputs,item_inputs_len)
                item_outputs, item_last_states=self._augru_rnn(item_input,gru_item_outputs,item_inputs_len,params)
            att_ouputs = tf.concat([user_id,item_input,item_last_states],1)


            with tf.variable_scope("fcn-net"):
                deep_input = att_ouputs
                for i in range(len(layers)):
                    deep_input = tf.layers.dense(deep_input,layers[i],activation=tf.nn.relu,kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg))
                    if mode == tf.estimator.ModeKeys.TRAIN:
                        deep_input = tf.layers.batch_normalization(deep_input,training = True)
                        deep_input = tf.nn.dropout(deep_input,dropout[i])
                    else:
                        deep_input = tf.layers.batch_normalization(deep_input,training = False)
            
            y_out = tf.layers.dense(deep_input,1)
            mian_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_out, labels=labels)) 

            with tf.variable_scope("aux-loss"):
                features_neg = self._neg_sampling(features['item_list'],mid_cat,neg_count)
                item_neg_inputs,item_neg_inputs_len = tf.contrib.feature_column.sequence_input_layer(features_neg,[feature_columns['item_list'],feature_columns['item_cat_list']])
                aux_pos_inputs = tf.concat([gru_item_outputs[:,:-1,:],item_inputs[:,1:,:]],axis=-1)
                tmp_gru_item_outputs = tf.tile(gru_item_outputs[:,:-1,:],[1,neg_count,1])
                aux_neg_inputs = tf.concat([tmp_gru_item_outputs,item_neg_inputs],axis=-1)
                pos_item_inputs_len = item_inputs_len -1
                y_pos_loss = self._auxiliary_loss(aux_pos_inputs,params,pos_item_inputs_len,True)
                y_neg_loss = self._auxiliary_loss(aux_neg_inputs,params,item_neg_inputs_len,False)
                aux_loss = (tf.reduce_sum(y_pos_loss) + 
                                tf.reduce_sum(y_neg_loss))/tf.cast((tf.reduce_sum(pos_item_inputs_len) + 
                                tf.reduce_sum(item_neg_inputs_len)),tf.float32)
                     
            prob = tf.sigmoid(y_out)
            loss = mian_loss + aux_loss + tf.losses.get_regularization_loss()

        return {"prob":prob,"loss":loss}