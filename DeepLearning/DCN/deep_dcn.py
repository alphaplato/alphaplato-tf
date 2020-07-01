#!/bin/python3
#coding:utf-8
#Copyright 2019 Alphaplato. All Rights Reserved.
#Desc:dcn model
#=======================================================
import tensorflow as tf

class DCN(object):
    def __init__(self,fg):
        self._model_name = 'DCN'
        self._fg = fg

    def build_logits(self,features,mode,params):
        feature_columns = self._fg.feature_columns
        dnn_layers = params['deep_layers']
        dcn_layers = params['dcn_layers']
        dropout = params['dropout']
        l2_reg = params['l2_reg']
        

        with tf.variable_scope("dcn_model"):
            deep_input = tf.feature_column.input_layer(features,feature_columns)
            with tf.variable_scope("dnn_net"):
                for i in range(len(dnn_layers)):
                    dnn_input = tf.layers.dense(deep_input,dnn_layers[i],activation=tf.nn.relu,kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg))
                    if mode == tf.estimator.ModeKeys.TRAIN:
                        dnn_input = tf.layers.batch_normalization(dnn_input,training = True)
                        dnn_input = tf.nn.dropout(dnn_input,dropout[i])
                    else:
                        dnn_input = tf.layers.batch_normalization(dnn_input,training = False)

            with tf.variable_scope("dcn_net"):
                xl = deep_input
                embed_dim = xl.get_shape().as_list()[1]
                for i in range(dcn_layers):
                    wl = tf.get_variable(name= "cross_w_{0}".format(i),shape=[embed_dim],regularizer=tf.contrib.layers.l2_regularizer(l2_reg),initializer=tf.truncated_normal_initializer(stddev=0.01))
                    bl = tf.get_variable(name= "cross_b_{0}".format(i),shape=[embed_dim],regularizer=tf.contrib.layers.l2_regularizer(l2_reg),initializer=tf.truncated_normal_initializer(stddev=0.01))
                    xlw = tf.matmul(xl,tf.reshape(wl,shape=[-1,1]))
                    xl = tf.multiply(deep_input,xlw) + bl + xl
                    # if mode == tf.estimator.ModeKeys.TRAIN:
                    #     xl = tf.layers.batch_normalization(xl,training = True)
                    # else:
                    #     xl = tf.layers.batch_normalization(xl,training = False)
            
            y_input = tf.concat([dnn_input,xl],axis=1)
            y_out = tf.layers.dense(xl,1)

            prob = tf.sigmoid(y_out)
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_out, labels=labels)) + tf.losses.get_regularization_loss()

        return {"prob":prob,"loss":loss}