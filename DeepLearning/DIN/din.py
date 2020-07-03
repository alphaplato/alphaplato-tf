#!/bin/python3
#coding:utf-8
#Copyright 2020 Alphaplato. All Rights Reserved.
#Desc:din model
#=======================================================

import tensorflow as tf

class DIN(object):
    def __init__(self,fg):
        self._model_name = 'DIN'
        self._fg = fg

    def _attention(self,queries,keys,keys_length,params):
        atten_layers = params['atten_layers']
        keys_shape = tf.shape(keys)
        queries = tf.tile(queries,[1,keys_shape[1]])
        queries = tf.reshape(queries,[keys_shape[0],keys_shape[1],keys.get_shape().as_list()[2]])
        # 将batch_size*keys_length当作batch处理，即在embedding层面做处理
        din_input = tf.concat([queries,keys,queries-keys],axis=-1)
        # print(keys_shape)
        for layer in atten_layers:
            din_input = tf.layers.dense(din_input, layer, activation=tf.nn.relu, name='fl_att_{0}'.format(layer))
        din_weight = tf.layers.dense(din_input, 1, activation=tf.nn.sigmoid, name='f1_att_weight')
        
        din_weight = tf.squeeze(din_weight)
        key_masks = tf.sequence_mask(keys_length,keys_shape[1])
        paddings = tf.ones_like(din_weight) / (2 ** 32 - 1) 
        din_weight = tf.where(key_masks,din_weight,paddings)
        din_weight = tf.nn.softmax(din_weight)
        din_weight = tf.expand_dims(din_weight,2)
        
        #attention action
        din_output = din_weight*keys
        din_output = tf.matmul(din_weight,keys,transpose_a=True)
        din_output = tf.reshape(din_output,[-1,keys.get_shape().as_list()[2]])

        return din_output      

    def build_logits(self,features,labels,params,mode=tf.estimator.ModeKeys.TRAIN):
        feature_columns = self._fg.feature_columns
        layers = params['deep_layers']
        dropout = params['dropout']
        l2_reg = params['l2_reg']

        with tf.variable_scope("din"):
            with tf.variable_scope("attention"):
                user_id = tf.feature_column.input_layer(features,feature_columns['user_id'])
                item_input = tf.feature_column.input_layer(features,[feature_columns['item_id'],feature_columns['item_cat']])
                #输出batch的embedings和batch内单个序列长度组成的list
                item_input_list,item_input_list_len = tf.contrib.feature_column.sequence_input_layer(features,[feature_columns['item_list'],feature_columns['item_cat_list']])
                with tf.variable_scope("attention-net"):
                    att_item_out = self._attention(item_input,item_input_list,item_input_list_len,params)
                att_concat = tf.concat([user_id,item_input,att_item_out],1)

            with tf.variable_scope("deep-net"):
                for i in range(len(layers)):
                    deep_input = tf.layers.dense(att_concat,layers[i],activation=tf.nn.relu,kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg))
                    if mode == tf.estimator.ModeKeys.TRAIN:
                        deep_input = tf.layers.batch_normalization(deep_input,training = True)
                        deep_input = tf.nn.dropout(deep_input,dropout[i])
                    else:
                        deep_input = tf.layers.batch_normalization(deep_input,training = False)
            
            y_out = tf.layers.dense(deep_input,1)
            prob = tf.sigmoid(y_out)
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_out, labels=labels)) + tf.losses.get_regularization_loss()

        return {"prob":prob,"loss":loss}