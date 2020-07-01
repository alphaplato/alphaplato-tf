#!/bin/python3
#coding:utf-8
#Copyright 2020 Alphaplato. All Rights Reserved.
#Desc:structed model
#=======================================================
import tensorflow as tf
import multiprocessing

MULTI_THREADING = True

class Model(object):
    def __init__(self,fg,md):
        self.fg = fg
        self.md = md

    def _parser(self,record):
        feature_spec = self.fg.feature_spec
        features = tf.parse_single_example(record, feature_spec)
        label = features.pop('label')
        return features,label

    #以下两种处理方式对于单数值特征列处理效果一样
    # fea_dict[fea['feature_name']] =  tf.FixedLenFeature(shape=[1],dtype=tf.string)
    # fea_dict[fea['feature_name']] = tf.VarLenFeature(tf.string)

    def input_fn(self,data_path,mode=tf.estimator.ModeKeys.TRAIN,batch_size=1,num_epochs=1):
        num_threads = multiprocessing.cpu_count() if MULTI_THREADING else 1
        dataset = tf.data.TFRecordDataset(data_path,num_parallel_reads=num_threads,compression_type='GZIP')
        dataset = dataset.map(self._parser).repeat(num_epochs).batch(batch_size)
        features,label = dataset.make_one_shot_iterator().get_next()
        return features,label

    def model_fn(self,features,labels,mode,params):
        optimizer = params['optimizer'] 
        learning_rate = params['learning_rate']

        output = self.md.build_logits(features,labels,params,mode)
        prob = output["prob"]
        predictions={"prob": prob}

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                export_outputs=export_outputs)

        export_outputs = {tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(predictions)}            
        
        eval_metric_ops = {
            "auc": tf.metrics.auc(labels, prob)
            }

        loss = output["loss"]
        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                loss=loss,
                eval_metric_ops=eval_metric_ops)

        if optimizer == 'Adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)
        elif optimizer == 'Adagrad':
            optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate, initial_accumulator_value=1e-8)
        elif optimizer == 'Momentum':
            optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.95)
        elif optimizer == 'ftrl':
            optimizer = tf.train.FtrlOptimizer(learning_rate)

        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

        if mode == tf.estimator.ModeKeys.TRAIN:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                loss=loss,
                train_op=train_op)