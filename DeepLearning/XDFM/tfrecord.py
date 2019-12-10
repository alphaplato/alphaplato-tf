#!/bin/python3 
#coding:utf-8
#author:alphaplato

import sys
import pandas as pd 
import tensorflow as tf

def tfrecords(infile,outfile):
    names = ['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','label']
    csv = pd.read_csv("data/"+infile,names = names).values
    csv['label'] = csv['label'].apply(lambda x : (x=='>50K'))
    with tf.python_io.TFRecordWriter("data/"+outfile+".tfrecords") as writer:
        for row in csv:
            features, label = row[:-1], row[-1]
            example = tf.train.Example()
            example.features.feature["features"].float_list.value.extend(features)
            example.features.feature["label"].int64_list.value.append(label)
            writer.write(example.SerializeToString())

if __name__=='__main__':
    tfrecords(sys.argv[1],sys.argv[2])