import json
import multiprocessing
import tensorflow as tf

class Model(object):
    def __init__(self,fg):
        self.fg = fg

    def build_logits(self,features,labels,mode,params):
        feature_columns =self.fg.feature_columns
        l2_reg = params["l2_reg"]
        layers = list(map(int, params["deep_layers"].split(',')))
        dropout = list(map(float, params["dropout"].split(',')))

        with tf.variable_scope("deepfm"):
            with tf.variable_scope("lr-part"):
                lr_feature_columns = [feature_columns['lr'][feature_name] for feature_name in feature_columns['lr']]
                lr_input = tf.feature_column.input_layer(features,lr_feature_columns)
                lr_out = tf.layers.dense(lr_input,1,kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg))
                if mode == tf.estimator.ModeKeys.TRAIN:
                    lr_out = tf.layers.batch_normalization(lr_out,training = True)
                else:
                    lr_out = tf.layers.batch_normalization(lr_out,training = False)

            with tf.variable_scope("fm-part"):
                feature_ids = [feature_columns['fm'][fea['feature_name']] for fea in self.fg._feature_json['features'] if fea['feature_type'] == 'id']
                ids_fields_size = len(feature_ids)
                ids_input = tf.feature_column.input_layer(features,feature_ids)

                raw_fields = [fea['feature_name'] for fea in self.fg._feature_json['features'] if fea['feature_type'] == 'raw']
                feature_raws = [feature_columns['fm'][feature_name] for feature_name in raw_fields]
                raws_value_input = tf.feature_column.input_layer(features,feature_raws)

                feature_raws_embed = {'raw_fields':raw_fields}
                raws_embed_input = tf.feature_column.input_layer(feature_raws_embed,[feature_columns['fm']['raw_fields']])
                
                embed_dim = raws_embed_input.get_shape().as_list()[1]
                raws_embed_transpose = tf.transpose(raws_embed_input)
                raws_embed_split = tf.split(raws_embed_transpose,embed_dim,axis=0)

                raws_input_m = tf.multiply(raws_value_input,raws_embed_split)
                raws_input_t = tf.transpose(raws_input_m,perm=[1,2,0])

                ids_input_s = tf.reshape(ids_input,[-1,ids_fields_size,embed_dim])
                fm_input = tf.concat([ids_input_s,raws_input_t],1)

                sum_square = tf.square(tf.reduce_sum(fm_input,axis=1))
                square_sum = tf.reduce_sum(tf.square(fm_input),axis=1)
                fm_out = 0.5*tf.reduce_sum(sum_square-square_sum,axis=1)        
                fm_out = tf.reshape(fm_out,[-1,1])

            with tf.variable_scope("deep-part"):
                kernel_regularizer = tf.contrib.layers.l2_regularizer(l2_reg)
                deep_feature_columns = [feature_columns['deep'][feature_name] for feature_name in feature_columns['deep']]
                deep_input = tf.feature_column.input_layer(features,deep_feature_columns)
                for i in range(len(layers)):
                    deep_input = tf.layers.dense(deep_input,layers[i],activation=tf.nn.relu,kernel_regularizer=kernel_regularizer)
                    if mode == tf.estimator.ModeKeys.TRAIN:
                        deep_input = tf.layers.batch_normalization(deep_input,training = True)
                        deep_input = tf.nn.dropout(deep_input,dropout[i])
                    else:
                        deep_input = tf.layers.batch_normalization(deep_input,training = False)
                deep_out = tf.layers.dense(deep_input,1,kernel_regularizer=kernel_regularizer)
            return lr_out,fm_out,deep_out


    def model_fn(self,features,labels,mode,params): 
        feature_columns =self.fg.feature_columns
        learning_rate = params["learning_rate"]
        optimizer = params["optimizer"]   

        lr_out,fm_out,deep_out = self.build_logits(features,labels,mode,params)
        y_out = lr_out +  fm_out + deep_out 
        
        labels = tf.cast(labels,tf.float32)
        pred = tf.sigmoid(y_out) 

        predictions={"prob": pred}
        export_outputs = {tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(predictions)} 
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                export_outputs=export_outputs)           
        
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_out, labels=labels)) + \
            tf.losses.get_regularization_loss()

        eval_metric_ops = {
            "auc": tf.metrics.auc(labels, pred)
            }
        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                loss=loss,
                eval_metric_ops=eval_metric_ops)

        if optimizer == 'Adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)
        elif optimizer == 'Adagrad':
            optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate, initial_accumulator_value=1e-8)
        elif optimizer == 'Momentum':
            optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.95)
        elif optimizer == 'ftrl':
            optimizer = tf.train.FtrlOptimizer(learning_rate)

        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

        if mode == tf.estimator.ModeKeys.TRAIN:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                loss=loss,
                train_op=train_op)