#!/bin/python3
#coding:utf-8
#Copyright 2019 Alphaplato. All Rights Reserved.
#Desc:reading tfrecord
#=======================================================
import json

def feature_json(fea_json_path):
    with open(fea_json_path,'r') as fr:
        fea_json = json.load(fr)
    return fea_json
