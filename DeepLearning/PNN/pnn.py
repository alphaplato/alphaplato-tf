#!/bin/python3
#coding:utf-8
#Copyright 2019 Alphaplato. All Rights Reserved.
#Desc:pnn model,inner product
#=======================================================
import tensorflow as tf

class PNN(object):
    def __init__(self,fg):
        self._model_name = 'PNN'
        self._fg = fg

    def build_logits(self,features,params,mode=tf.estimator.ModeKeys.TRAIN):
        feature_columns = self._fg.feature_columns
        layers = params['deep_layers']
        dropout = params['dropout']
        l2_reg = params['l2_reg']

        with tf.variable_scope("pnn"):
            with tf.variable_scope("pnn-inner-part-in"):
                feature_ids = [feature_columns['fm'][fea['feature_name']] for fea in self._fg._feature_json['features'] if fea['feature_type'] == 'id']
                ids_fields_size = len(feature_ids)
                ids_input = tf.feature_column.input_layer(features,feature_ids)

                raw_fields = [fea['feature_name'] for fea in self._fg._feature_json['features'] if fea['feature_type'] == 'raw']
                raw_fields_size = len(raw_fields)
                feature_raws = [feature_columns['fm'][feature_name] for feature_name in raw_fields]
                raws_value_input = tf.feature_column.input_layer(features,feature_raws)

                feature_raws_embed = {'raw_fields':raw_fields}
                raws_embed_input = tf.feature_column.input_layer(feature_raws_embed,[feature_columns['fm']['raw_fields']])
                
                embed_dim = raws_embed_input.get_shape().as_list()[1]
                raws_embed_transpose = tf.transpose(raws_embed_input)
                raws_embed_split = tf.split(raws_embed_transpose,embed_dim,axis=0)

                raws_input_m = tf.multiply(raws_value_input,raws_embed_split)
                raws_input_t = tf.transpose(raws_input_m,perm=[1,2,0])

                ids_input_s = tf.reshape(ids_input,[-1,ids_fields_size,embed_dim])
                fm_input = tf.concat([ids_input_s,raws_input_t],1)

                pnn_input = tf.split(fm_input,embed_dim,axis=2)
                pnn_input = tf.multiply(pnn_input,tf.transpose(pnn_input,perm=[0,1,3,2]))
                pnn_input = tf.reduce_sum(pnn_input,axis=0)
                fields_size_square = (ids_fields_size+raw_fields_size)*(ids_fields_size+raw_fields_size)
                pnn_input = tf.reshape(pnn_input,[-1,fields_size_square])         

            with tf.variable_scope("deep-part-in"):
                deep_feature_columns = [feature_columns['deep'][feature_name] for feature_name in feature_columns['deep']]
                deep_input = tf.feature_column.input_layer(features,deep_feature_columns)

            with tf.variable_scope("deep-out-part"):
                x_input = tf.concat([pnn_input,deep_input],1)
                for i in range(len(layers)):
                    x_input = tf.layers.dense(x_input,layers[i],activation=tf.nn.relu,kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg))
                    if mode == tf.estimator.ModeKeys.TRAIN:
                        x_input = tf.layers.batch_normalization(x_input,training = True)
                        x_input = tf.nn.dropout(x_input,dropout[i])
                    else:
                        x_input = tf.layers.batch_normalization(x_input,training = False)

            y_out = tf.layers.dense(x_input,1)

        return y_out