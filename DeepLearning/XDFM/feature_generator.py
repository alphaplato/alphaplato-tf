#!/bin/python3
#coding:utf-8
#desc:feature generate

import tensorflow as tf
import json

class Feature_Column:
    def __init__(self,fg):
        self.fg = fg
    
    def __fg__(self):
        with open(self.fg,'r') as fr:
            fg = json.load(fr)
        
        ids = [elem["feature_name"] for elem in fg]
        raw = [elem["feature_name"] for elem in fg]

        feature_columns = []
        for fea in fg['features']:
            if fea['type'] == 'id':   
                ids = tf.feature_column.categorical_column_with_hash_bucket(id,hash_bucket_size=fea['hash_size'])
            elif fea['type'] == 'id':
                fea = tf
