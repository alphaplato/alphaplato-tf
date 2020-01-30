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

    def _feature_generate(self):
        feature_columns = {}     
        #feature columns process
        for fea in self._feature_json['features']:
            if fea['feature_type'] == 'id':
                x_feature = tf.feature_column.categorical_column_with_hash_bucket(fea['feature_name'],hash_bucket_size=fea['hash_size'])
                feature_columns[fea['feature_name']] = tf.feature_column.embedding_column(x_feature,fea['embedding'])
            elif fea['feature_type'] == 'list':
                x_feature = tf.feature_column.sequence_categorical_column_with_hash_bucket(fea['feature_name'],hash_bucket_size=fea['hash_size'])
                feature_columns[fea['feature_name']] = tf.feature_column.embedding_column(x_feature,fea['embedding'])
        self.feature_columns = feature_columns