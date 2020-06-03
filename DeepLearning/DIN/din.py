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

    def attention(self,queries,keys,keys_length,params):
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
        paddings = tf.ones_like(din_weight) * (-2 ** 32 + 1) 
        din_weight = tf.where(key_masks,din_weight,paddings)
        din_weight = tf.nn.softmax(din_weight)
        din_weight = tf.expand_dims(din_weight,2)
        
        #attention action
        din_output = din_weight*keys
        din_output = tf.matmul(din_weight,keys,transpose_a=True)
        din_output = tf.reshape(din_output,[-1,keys.get_shape().as_list()[2]])

        return din_output      

    def build_logits(self,features,mode,params):
        feature_columns = self._fg.feature_columns
        layers = params['deep_layers']
        dropout = params['dropout']
        l2_reg = params['l2_reg']

        with tf.variable_scope("din"):
            with tf.variable_scope("attention"):
                item_id = tf.feature_column.input_layer(features,feature_columns['item_id'])
                item_cat = tf.feature_column.input_layer(features,feature_columns['item_cat'])
                #输出batch的embedings和batch内单个序列长度组成的list
                item_id_list,item_id_list_len = tf.contrib.feature_column.sequence_input_layer(features,feature_columns['item_list'])
                item_cat_list,item_cat_list_len = tf.contrib.feature_column.sequence_input_layer(features,feature_columns['item_cat_list'])
                with tf.variable_scope("attention-id"):
                    att_item_id = self.attention(item_id,item_id_list,item_id_list_len,params)
                with tf.variable_scope("attention-cat"):    
                    att_item_cat = self.attention(item_cat,item_cat_list,item_cat_list_len,params)
                att_concat = tf.concat([item_id,item_cat,att_item_id,att_item_cat],1)

            with tf.variable_scope("deep-net"):
                for i in range(len(layers)):
                    deep_input = tf.layers.dense(att_concat,layers[i],activation=tf.nn.relu,kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg))
                    if mode == tf.estimator.ModeKeys.TRAIN:
                        deep_input = tf.layers.batch_normalization(deep_input,training = True)
                        deep_input = tf.nn.dropout(deep_input,dropout[i])
                    else:
                        deep_input = tf.layers.batch_normalization(deep_input,training = False)
            
            y_out = tf.layers.dense(deep_input,1)

            return  y_out