#!/bin/python3
#coding:utf-8
#Copyright 2019 Alphaplato. All Rights Reserved.
#Desc:deepfm model
#=======================================================

import tensorflow as tf

class DeepFM(object):
    def __init__(self,fg):
        self._model_name = 'DeepFM'
        self._fg = fg

    def build_logits(self,features,labels,mode,params):
        layers = params['deep_layers']
        dropout = params['dropout']
        l2_reg = params['l2_reg']

        feature_columns = self._fg.feature_columns
        with tf.variable_scope("deepfm"):
            with tf.variable_scope("lr-part"):
                lr_feature_columns = [feature_columns['lr'][feature_name] for feature_name in feature_columns['lr']]
                lr_input = tf.feature_column.input_layer(features,lr_feature_columns)
                lr_out = tf.layers.dense(lr_input,1,kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg))
                if mode == tf.estimator.ModeKeys.TRAIN:
                    lr_out = tf.layers.batch_normalization(lr_out,training = True)
                else:
                    lr_out = tf.layers.batch_normalization(lr_out,training = False)

            with tf.variable_scope("fm-part"):
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

                sum_square = tf.square(tf.reduce_sum(fm_input,axis=1))
                square_sum = tf.reduce_sum(tf.square(fm_input),axis=1)
                fm_out = 0.5*tf.reduce_sum(sum_square-square_sum,axis=1)        
                fm_out = tf.reshape(fm_out,[-1,1])

            with tf.variable_scope("deep-part"):
                deep_feature_columns = [feature_columns['deep'][feature_name] for feature_name in feature_columns['deep']]
                deep_input = tf.feature_column.input_layer(features,deep_feature_columns)
                for i in range(len(layers)):
                    deep_input = tf.layers.dense(deep_input,layers[i],activation=tf.nn.relu,kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg))
                    if mode == tf.estimator.ModeKeys.TRAIN:
                        deep_input = tf.layers.batch_normalization(deep_input,training = True)
                        deep_input = tf.nn.dropout(deep_input,dropout[i])
                    else:
                        deep_input = tf.layers.batch_normalization(deep_input,training = False)
                deep_out = tf.layers.dense(deep_input,1) # 注意：输出层使用线性激活函数！！！！
            
                y_input = tf.concat([fm_out,deep_out],axis=1)
                y_out = tf.layers.dense(y_input,1)

            prob = tf.sigmoid(y_out)
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_out, labels=labels)) + tf.losses.get_regularization_loss()

        return {"prob":prob,"loss":loss}