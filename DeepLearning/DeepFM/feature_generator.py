#!/bin/python3
#coding:utf-8
#Copyright 2019 Alphaplato. All Rights Reserved.
#Desc:read tfrecord and generate feature columns
#=======================================================
import json
import multiprocessing
import tensorflow as tf

MULTI_THREADING = True

def feature_json(fea_json_path):
    with open(fea_json_path,'r') as fr:
        fea_json = json.load(fr)
    return fea_json

class FeatureGenerator(object):
    def __init__(self,feature_json):
        self._feature_json = feature_json
        self._feature_generate()

    def _feature_generate(self):
        feature_columns = {}
        #deepfm lr feature columns process
        lr_feature_columns = {}
        for fea in self._feature_json['features']:
            if fea['feature_type'] == 'raw':
                lr_feature_columns[fea['feature_name']] = tf.feature_column.numeric_column(fea['feature_name'])
            elif fea['feature_type'] == 'id':
                x_feature = tf.feature_column.categorical_column_with_hash_bucket(fea['feature_name'],hash_bucket_size=fea['hash_size'])
                lr_feature_columns[fea['feature_name']] = tf.feature_column.embedding_column(x_feature,dimension=1)
        feature_columns['lr'] = lr_feature_columns
        #deepfm fm feature columns process
        fm_feature_columns = {}
        for fea in self._feature_json['features']:
            if fea['feature_type'] == 'raw':
                fm_feature_columns[fea['feature_name']] = tf.feature_column.numeric_column(fea['feature_name'])
            elif fea['feature_type'] == 'id':
                x_feature = tf.feature_column.categorical_column_with_hash_bucket(fea['feature_name'],hash_bucket_size=fea['hash_size'])
                fm_feature_columns[fea['feature_name']] = tf.feature_column.embedding_column(x_feature,dimension=fea['embedding'])
            raw_fields = [fea['feature_name'] for fea in self._feature_json['features']]
            x_feature = tf.feature_column.categorical_column_with_vocabulary_list('raw_fields',raw_fields)
            fm_feature_columns['raw_fields'] = tf.feature_column.embedding_column(x_feature,dimension=16)
          
        feature_columns['fm'] = fm_feature_columns       
        #deepfm deep feature columns process
        deep_feature_columns = {}
        for fea in self._feature_json['features']:
            if fea['feature_type'] == 'raw':
                deep_feature_columns[fea['feature_name']] = tf.feature_column.numeric_column(fea['feature_name'])
            elif fea['feature_type'] == 'id':
                x_feature = tf.feature_column.categorical_column_with_hash_bucket(fea['feature_name'],hash_bucket_size=fea['hash_size'])
                deep_feature_columns[fea['feature_name']] = tf.feature_column.embedding_column(x_feature,dimension=fea['embedding'])
        feature_columns['deep'] = deep_feature_columns
        self.feature_columns = feature_columns

    def _parser(self,record):
        fea_dict = {}
        for fea in self._feature_json['features']:
            if fea['value_type'] == 'Double':
                fea_dict[fea['feature_name']] = tf.FixedLenFeature(shape=[1],dtype=tf.float32)
            elif fea['value_type'] == 'String':
                fea_dict[fea['feature_name']] = tf.VarLenFeature(tf.string)
        fea_dict['label'] = tf.FixedLenFeature(shape=[1],dtype=tf.int64)
        features = tf.parse_single_example(record, fea_dict)
        label =  features.pop('label')
        return features,label

    def input_fn(self,data_path,mode=tf.estimator.ModeKeys.TRAIN,batch_size=1,num_epochs=1):
        num_threads = multiprocessing.cpu_count() if MULTI_THREADING else 1
        dataset = tf.data.TFRecordDataset(data_path,num_parallel_reads=num_threads)
        dataset = dataset.map(self._parser).repeat(num_epochs).batch(batch_size)
        features,label = dataset.make_one_shot_iterator().get_next()
        return features,label