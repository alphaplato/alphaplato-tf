from __future__ import print_function

# This is a placeholder for a Google-internal import.

import grpc
import requests
import tensorflow.compat.v1 as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc


tf.app.flags.DEFINE_string('server', 'localhost:8500',
                           'PredictionService host:port')
tf.app.flags.DEFINE_string('image', '', 'path to image in JPEG format')
FLAGS = tf.app.flags.FLAGS


def main(_):
  samples={"age":[37.0,37.0],"workclass":["Private","Private"],"education":["Some-college","Some-college"],"education-num":["10","10"],"marital-status":["Married-civ-spouse","Married-civ-spouse"],"occupation":["Exec-managerial","Exec-managerial"],"relationship":["Husband","Husband"],"race":["Black","Black"],"sex":["Male","Male"],"capital-gain":[0.0,10.0],"capital-loss":[0.0,100.0],"hours-per-week":[80.0,45.0],"native-country":["United-States","United-States"]}
  channel = grpc.insecure_channel(FLAGS.server)
  stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
  # Send request
  # See prediction_service.proto for gRPC request/response details.
  request = predict_pb2.PredictRequest()
  request.model_spec.name = 'deepfm'
  request.model_spec.signature_name = 'serving_default'
  for key,value in samples.items():
    request.inputs[key].CopyFrom(
      tf.make_tensor_proto(value, shape=[2]))
  result = stub.Predict(request, 10.0)  # 10 secs timeout
  print(result)

if __name__ == '__main__':
  tf.compat.v1.app.run()