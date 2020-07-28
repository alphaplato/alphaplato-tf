#!/usr/bin/env python
# coding=utf-8
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import shutil
import os
import json
import tensorflow as tf

from common import feature_json
from feature_generator import FeatureGenerator
from din import DIN
from model import Model

#################### CMD Arguments ####################
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("dist_mode", 0, "distribuion mode {0-loacal, 1-single_dist, 2-multi_dist}")
tf.app.flags.DEFINE_string("ps_hosts", 'localhost:2222', "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", 'localhost:2223,localhost:2224', "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("job_name", '', "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
tf.app.flags.DEFINE_integer("num_threads", 16, "Number of threads")
tf.app.flags.DEFINE_integer("num_epochs", 20, "Number of epochs")
tf.app.flags.DEFINE_integer("batch_size", 32, "Number of batch size")
tf.app.flags.DEFINE_integer("log_steps", 1000, "save summary every steps")
tf.app.flags.DEFINE_integer("save_checkpoints_secs", 10, "save checkpoints every seconds)
tf.app.flags.DEFINE_float("learning_rate", 0.0005, "learning rate")
tf.app.flags.DEFINE_float("l2_reg", 0.01, "L2 regularization")
tf.app.flags.DEFINE_string("optimizer", 'Adam', "optimizer type {Adam, Adagrad, GD, Momentum}")
tf.app.flags.DEFINE_string("deep_layers", '20,10,5', "deep layers")
tf.app.flags.DEFINE_string("atten_layers", '32,16', "xdeep layers")
tf.app.flags.DEFINE_string("dropout", '0.7,0.7,0.5', "dropout rate")
tf.app.flags.DEFINE_string("data_dir", '', "data dir")
tf.app.flags.DEFINE_string("dt_dir", '', "data dt partition")
tf.app.flags.DEFINE_string("model_dir", '', "model check point dir")
tf.app.flags.DEFINE_string("servable_model_dir", 'export', "export servable model for TensorFlow Serving")
tf.app.flags.DEFINE_string("task_type", 'train', "task type {train, infer, eval, export}")
tf.app.flags.DEFINE_boolean("clear_existing_model", False, "clear existing model or not")


def set_dist_env():
    if FLAGS.dist_mode == 1:  # 本地分布式测试模式1 chief, 1 ps, 1 evaluator
        ps_hosts = FLAGS.ps_hosts.split(',')
        worker_hosts = FLAGS.worker_hosts.split(',')
        chief_hosts = worker_hosts[0:1]  # get first worker as chief
        worker_hosts = worker_hosts[1:]
        task_index = FLAGS.task_index
        job_name = FLAGS.job_name
        print('ps_host', ps_hosts)
        print('chief_hosts', chief_hosts)
        print('job_name', job_name)
        print('task_index', str(task_index))
        # 无worker参数
        tf_config = {
            'cluster': {'chief': chief_hosts, 'worker': worker_hosts, 'ps': ps_hosts},
            'task': {'type': job_name, 'index': task_index}
        }
        print(json.dumps(tf_config))
        os.environ['TF_CONFIG'] = json.dumps(tf_config)
    elif FLAGS.dist_mode == 2:  # 集群分布式模式
        ps_hosts = FLAGS.ps_hosts.split(',')
        worker_hosts = FLAGS.worker_hosts.split(',')
        chief_hosts = worker_hosts[0:1]  # get first worker as chief
        worker_hosts = worker_hosts[1:]  # the rest as worker
        task_index = FLAGS.task_index
        job_name = FLAGS.job_name
        print('ps_host', ps_hosts)
        print('worker_host', worker_hosts)
        print('chief_hosts', chief_hosts)
        print('job_name', job_name)
        print('task_index', str(task_index))
        # use #worker=0 as chief
        if job_name == "worker" and task_index == 0:
            job_name = "chief"
        # use #worker=1 as evaluator
        if job_name == "worker" and task_index == 1:
            job_name = 'evaluator'
            task_index = 0
        # the others as worker
        if job_name == "worker" and task_index > 1:
            task_index -= 2

        tf_config = {
            'cluster': {'chief': chief_hosts, 'worker': worker_hosts, 'ps': ps_hosts},
            'task': {'type': job_name, 'index': task_index}
        }
        print(json.dumps(tf_config))
        os.environ['TF_CONFIG'] = json.dumps(tf_config)

def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    set_dist_env()
    #------bulid Tasks------
    model_params = {
        "learning_rate": FLAGS.learning_rate,
        "l2_reg": FLAGS.l2_reg,
        "deep_layers": list(map(int,FLAGS.deep_layers.split(','))),
        "atten_layers":list(map(int,FLAGS.atten_layers.split(','))),
        "dropout": list(map(float,FLAGS.dropout.split(','))),
        "optimizer":FLAGS.optimizer
    }
  
    tr_files = "./data/train.tfrecords"
    va_files ="./data/test.tfrecords"

    fea_json = feature_json('./feature_generator.json')
    fg = FeatureGenerator(fea_json)
    md = DIN(fg)
    model = Model(fg,md)

    config = tf.estimator.RunConfig().replace(
        session_config = tf.ConfigProto(device_count={'GPU':0, 'CPU':FLAGS.num_threads}),
        log_step_count_steps=FLAGS.log_steps,
        save_summary_steps=FLAGS.log_steps,
        save_checkpoints_secs=FLAGS.save_checkpoints_secs)
    Estimator = tf.estimator.Estimator(
        model_fn=model.model_fn,
        model_dir='./model/',
        params=model_params,
        config=config)

    if FLAGS.task_type == 'train':
        train_spec = tf.estimator.TrainSpec(
            input_fn=lambda: model.input_fn(
                tr_files, 
                num_epochs=FLAGS.num_epochs,
                batch_size=FLAGS.batch_size))
        eval_spec = tf.estimator.EvalSpec(
            input_fn=lambda: model.input_fn(
                va_files, 
                num_epochs=1, 
                batch_size=FLAGS.batch_size),
            steps=None, 
            start_delay_secs=10, 
            throttle_secs=FLAGS.save_checkpoints_secs)
        tf.estimator.train_and_evaluate(
            Estimator,
            train_spec,
            eval_spec)
    elif FLAGS.task_type == 'eval':
        Estimator.evaluate(
            input_fn=lambda: model.input_fn(
                tr_files, 
                num_epochs=1, 
                batch_size=FLAGS.batch_size))
        Estimator.evaluate(
            input_fn=lambda: model.input_fn(
                va_files, 
                num_epochs=1, 
                batch_size=FLAGS.batch_size))
    elif FLAGS.task_type == 'infer':
        preds = Estimator.predict(
            input_fn=lambda: model.input_fn(
                va_files, 
                num_epochs=1, 
                batch_size=FLAGS.batch_size), 
            predict_keys="prob")
    elif FLAGS.task_type == 'export':
        ##单机使用保存
        # print(fg.feature_spec)
        # serving_input_receiver_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(fg.feature_spec)
        serving_input_receiver_fn = (
            tf.estimator.export.build_raw_serving_input_receiver_fn(fg.feature_placeholders)
        )

        Estimator.export_saved_model(
            FLAGS.servable_model_dir,
            serving_input_receiver_fn)

if __name__=='__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()