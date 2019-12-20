#!/bin/python3
#coding:utf-8
#Copyright 2019 Alphaplato. All Rights Reserved.
#Desc:make tfrecord from csv
#=======================================================

import pandas as pd
import tensorflow as tf
import json

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
        names=['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','label']
        csv = pd.read_csv(self.csv_fin_path,names=names,skiprows=1)
        csv.label = csv.label.apply(lambda x: int('>' in x.strip()))
        csv['capital-gain'] =  csv['capital-gain'].apply(lambda x: math.log(x+1))
        csv['capital-loss'] =  csv['capital-loss'].apply(lambda x: math.log(x+1))

        csv = csv.values
        self._feature_json()
        with tf.python_io.TFRecordWriter(self.fout_tfrecord) as writer:
            for row in csv:
                features, label = row[:-1], row[-1]
                fea_dict = dict(zip(names,features))
                feature = {}
                for fea in self.fea_list:
                    if fea['feature_type'] == 'raw':
                        feature[fea['feature_name']] = tf.train.Feature(float_list=tf.train.FloatList(value=[float(fea_dict[fea['feature_name']])]))
                    elif fea['feature_type'] == 'id':
                        feature[fea['feature_name']] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(fea_dict[fea['feature_name']]).encode()]))
                feature['label'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
                example = tf.train.Example(features = tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())

if __name__=='__main__':
    CsvTfrecord('./data/adult.data','./feature_generator.json','./data/train.tfrecords')
    CsvTfrecord('./data/adult.test','./feature_generator.json','./data/test.tfrecords')