#!/bin/python3
#coding:utf-8
#Copyright 2020 Alphaplato. All Rights Reserved.
#Desc:reading tfrecord

from __future__ import print_function

import base64
import requests
import json

# fill your ip and port
SERVER_URL = 'http://172.16.16.6:30816/v1/models/deepfm:predict'

def main():
  single_sample = {"age":37.0,"workclass":"Private","education":"Some-college","education-num":"10","marital-status":"Married-civ-spouse","occupation":"Exec-managerial","relationship":"Husband","race":"Black","sex":"Male","capital-gain":0.0,"capital-loss":0.0,"hours-per-week":80.0,"native-country":"United-States"} 
  many_samples = [single_sample] * 1000
  #predict_request = '{"instances" : [%s]}' % single_sample
  predict_request = '{"instances" :  %s}' % json.dumps(many_samples)
  # Send few requests to warm-up the model.
  for _ in range(1):
    response = requests.post(SERVER_URL, data=predict_request)
    response.raise_for_status()

  # Send few actual requests and report average latency.
  total_time = 0
  num_requests = 1000
  for _ in range(num_requests):
    response = requests.post(SERVER_URL, data=predict_request)
    response.raise_for_status()
    total_time += response.elapsed.total_seconds()
    prediction = response.json()['predictions'][0]
  print('Prediction score: {}, avg latency: {} ms'.format(
      prediction[0], (total_time*1000)/num_requests))

if __name__ == '__main__':
  main()