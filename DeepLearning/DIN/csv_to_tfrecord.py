#!/bin/python3
#coding:utf-8
#Copyright 2019 Alphaplato. All Rights Reserved.
#Desc:make tfrecord from csv
#=======================================================

import pandas as pd
import tensorflow as tf
import json
import math

class CsvTfrecord(object):
    def __init__(self,csv_fin_path,fea_json_path,fout_tfrecord):
        self.csv_fin_path = csv_fin_path
        self.fea_json_path = fea_json_path
        self.fout_tfrecord = fout_tfrecord
        self._mk_tfrecord()

    def _feature_json(self):
        with open(self.fea_json_path,'r') as fr:
            fea_json = json.load(fr)
            fea_list = fea_json['features']
            self.fea_list = fea_list

    def _mk_tfrecord(self):
        names=['label','user_id','item_id','item_cat','item_list','item_cat_list']
        csv = pd.read_csv(self.csv_fin_path,names=names,sep='\t')
        csv.label = csv.label.apply(int)
        csv['item_list'] =  csv['item_list'].apply(lambda x: x.encode().split(b'\x02'))
        csv['item_cat_list'] =  csv['item_cat_list'].apply(lambda x: x.encode().split(b'\x02'))
        pd.set_option('display.max_columns',None)
        # print(csv.head(5))

        csv = csv.values
        self._feature_json()
        options_gzip = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
        with tf.python_io.TFRecordWriter(self.fout_tfrecord,options=options_gzip) as writer:
            for row in csv:
                fea_dict = dict(zip(names,row))
                feature = {}
                for fea in self.fea_list:
                    if fea['feature_type'] == 'list':
                        feature[fea['feature_name']] = tf.train.Feature(bytes_list=tf.train.BytesList(value=fea_dict[fea['feature_name']]))
                    elif fea['feature_type'] == 'id':
                        feature[fea['feature_name']] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(fea_dict[fea['feature_name']]).encode()]))
                feature['label'] = tf.train.Feature(float_list=tf.train.FloatList(value=[fea_dict['label']]))
                example = tf.train.Example(
                    features=tf.train.Features(feature=feature)
                )
                writer.write(example.SerializeToString())

if __name__=='__main__':
    CsvTfrecord('./data/local_train_splitByUser','./feature_generator.json','./data/train.tfrecords')
    CsvTfrecord('./data/local_test_splitByUser','./feature_generator.json','./data/test.tfrecords')