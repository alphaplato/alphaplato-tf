#!/bin/python3
#coding:utf-8
#Copyright 2019 Alphaplato. All Rights Reserved.
#Desc:xdeepfm model
#=======================================================
import tensorflow as tf

class XDFM(object):
    def __init__(self,fg):
        self._model_name = 'XDFM'
        self._fg = fg

    def build_logits(self,features,mode,params):
        feature_columns = self._fg.feature_columns
        layers = params['deep_layers']
        dropout = params['dropout']
        l2_reg = params['l2_reg']
        xdeep_layers = params['xdeep_layers']

        with tf.variable_scope("xdfm-net"):
            with tf.variable_scope("lr-part"):
                lr_feature_columns = [feature_columns['lr'][feature_name] for feature_name in feature_columns['lr']]
                lr_input = tf.feature_column.input_layer(features,lr_feature_columns)
                lr_out = tf.layers.dense(lr_input,1,kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg))
                # if mode == tf.estimator.ModeKeys.TRAIN:
                #     lr_out = tf.layers.batch_normalization(lr_out,training = True)
                # else:
                #     lr_out = tf.layers.batch_normalization(lr_out,training = False)

            with tf.variable_scope("xd-part"):
                #提取特征，这部分较为复杂，主要是因为连续特征采用了embedding的方式，因此连续特征处理两次
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
                
                # 正式进入XdeepFM的网络生成环节
                hid_layers = [fm_input]
                hid_fileds = [ids_fields_size+raw_fields_size]

                for i in range(len(xdeep_layers)):
                    split_0 = tf.split(hid_layers[0],embed_dim,axis=2)
                    split_i = tf.split(hid_layers[i],embed_dim,axis=2)
                    split_i = tf.transpose(split_i,perm=[0,1,3,2])
                    split_dot = tf.multiply(split_0,split_i)
                    reshape_size = hid_fileds[0]*hid_fileds[-1]
                    split_reshape = tf.reshape(tf.transpose(split_dot,perm=[1,0,2,3]),[-1,reshape_size,1])
                    split_conv = tf.layers.conv1d(split_reshape,filters=xdeep_layers[i],kernel_size=[reshape_size],strides=[1],padding='VALID')
                    hid_next = tf.reshape(split_conv,[-1,xdeep_layers[i],embed_dim])
                    hid_layers.append(hid_next)
                    hid_fileds.append(xdeep_layers[i])
                xdfm_input = tf.concat(hid_layers,axis=1) 
                xdfm_out = tf.reduce_sum(xdfm_input,2)      

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
                deep_out = deep_input

            x_input = tf.concat([lr_out,deep_out,xdfm_out],axis=1)
            # x_input = deep_out
            y_out = tf.layers.dense(x_input,1)
            prob = tf.sigmoid(y_out)
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_out, labels=labels)) + tf.losses.get_regularization_loss()

        return {"prob":prob,"loss":loss}