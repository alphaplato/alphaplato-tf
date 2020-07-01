#!/bin/python3
#coding:utf-8
#Copyright 2019 Alphaplato. All Rights Reserved.
#Desc:read tfrecord and generate feature columns
#=======================================================
import tensorflow as tf

class FeatureGenerator(object):
    def __init__(self,feature_json):
        self._feature_json = feature_json
        self._feature_generate()
        self._feature_specify()
        self._feature_holder()

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

    def _feature_specify(self):
        feature_spec = {}
        for fea in self._feature_json['features']:
            if fea['value_type'] == 'Double':
                feature_spec[fea['feature_name']] = tf.FixedLenFeature(shape=[1],dtype=tf.float32)
            elif fea['value_type'] == 'String':
                feature_spec[fea['feature_name']] = tf.VarLenFeature(tf.string)
        feature_spec['label'] = tf.FixedLenFeature(shape=[1],dtype=tf.float32)
        self.feature_spec = feature_spec

    def _feature_holder(self):
        feature_placeholders = {} # 支持build_raw_serving_input_receiver_fn（原始数据输入serving）使用
        for fea in self._feature_json['features']:
            if fea['feature_type'] == 'raw':
                feature_placeholders[fea['feature_name']] =  tf.placeholder(tf.float32,[1],name=fea['feature_name'])
            elif fea['feature_type'] == 'id':
                feature_placeholders[fea['feature_name']] =  tf.placeholder(tf.string,[1],name=fea['feature_name'])
        self.feature_placeholders = feature_placeholders