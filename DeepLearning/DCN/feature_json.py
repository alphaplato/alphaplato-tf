import pandas as pd
import tensorflow as tf
import json

def feature_json(fea_json_path):
    with open(fea_json_path,'r') as fr:
        fea_json = json.load(fr)
    return fea_json