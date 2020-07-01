#!/bin/python3
#coding:utf-8
#Copyright 2020 Alphaplato. All Rights Reserved.
#Desc:rnncell class
#=======================================================
from tensorflow.python.util.tf_export import tf_export
from tensorflow.python.util import nest
from tensorflow.python.ops.rnn_cell import LayerRNNCell
from tensorflow.python.eager import context
# from tensorflow.python.layers import base as base_layer
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.keras.engine import input_spec
from tensorflow.python.keras import initializers

_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"

@tf_export(v1=["nn.rnn_cell.AUGRUCell"])
# class AUGRUCell(RNNCell):
class AUGRUCell(LayerRNNCell):
# class AUGRUCell(base_layer.Layer):
    def __init__(self,
                 num_units,
                 activation=None,
                 reuse=None,
                 kernel_initializer=None,
                 bias_initializer=None,
                 name=None,
                 dtype=None,
                 **kwargs):
        super(AUGRUCell, self).__init__(
            _reuse=reuse, name=name, dtype=dtype, **kwargs)
        # _check_supported_dtypes(self.dtype)

        if context.executing_eagerly() and context.num_gpus() > 0:
            logging.warn(
                "%s: Note that this cell is not optimized for performance. "
                "Please use tf.contrib.cudnn_rnn.CudnnGRU for better "
                "performance on GPU.", self)
        # Inputs must be 2-dimensional.
        self.input_spec = input_spec.InputSpec(ndim=2)

        self._num_units = num_units
        if activation:
            self._activation = activations.get(activation)
        else:
            self._activation = math_ops.tanh
        self._kernel_initializer = initializers.get(kernel_initializer)
        self._bias_initializer = initializers.get(bias_initializer)

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    @tf_utils.shape_type_conversion
    def build(self, inputs_shape):
        if inputs_shape[-1] is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s" %
                             str(inputs_shape))
        # _check_supported_dtypes(self.dtype)
        input_depth = inputs_shape[-1]
        self._gate_kernel = self.add_variable(
            "gates/%s" % _WEIGHTS_VARIABLE_NAME,
            shape=[input_depth + self._num_units, 2 * self._num_units],
            initializer=self._kernel_initializer)
        self._gate_bias = self.add_variable(
            "gates/%s" % _BIAS_VARIABLE_NAME,
            shape=[2 * self._num_units],
            initializer=(self._bias_initializer
                         if self._bias_initializer is not None else
                         init_ops.constant_initializer(1.0, dtype=self.dtype)))
        self._candidate_kernel = self.add_variable(
            "candidate/%s" % _WEIGHTS_VARIABLE_NAME,
            shape=[input_depth + self._num_units, self._num_units],
            initializer=self._kernel_initializer)
        self._candidate_bias = self.add_variable(
            "candidate/%s" % _BIAS_VARIABLE_NAME,
            shape=[self._num_units],
            initializer=(self._bias_initializer
                         if self._bias_initializer is not None else
                         init_ops.zeros_initializer(dtype=self.dtype)))
        self.built = True

    def __call__(self, inputs, state, att_score=None):
        self._maybe_build(inputs)
        return self.call(inputs, state, att_score)

    def call(self, inputs, state, att_score=None):
        # """Gated recurrent unit (GRU) with nunits cells."""
        # _check_rnn_cell_input_dtypes([inputs, state])

        gate_inputs = math_ops.matmul(
            array_ops.concat([inputs, state], 1), self._gate_kernel)
        gate_inputs = nn_ops.bias_add(gate_inputs, self._gate_bias)

        value = math_ops.sigmoid(gate_inputs)
        r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)

        r_state = r * state

        candidate = math_ops.matmul(
            array_ops.concat([inputs, r_state], 1), self._candidate_kernel)
        candidate = nn_ops.bias_add(candidate, self._candidate_bias)

        c = self._activation(candidate)
        ## attention action
        if att_score is not None:
            u = (1.0 - att_score) * u
        new_h = u * state + (1 - u) * c
        return new_h, new_h

    def get_config(self):
        config = {
            "num_units": self._num_units,
            "kernel_initializer": initializers.serialize(self._kernel_initializer),
            "bias_initializer": initializers.serialize(self._bias_initializer),
            "activation": activations.serialize(self._activation),
            "reuse": self._reuse,
        }
        base_config = super(GRUCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def _maybe_build(self, inputs):
        # Check input assumptions set before layer building, e.g. input rank.
        if not self.built:
            input_spec.assert_input_compatibility(
                self.input_spec, inputs, self.name)
            input_list = nest.flatten(inputs)

            input_shapes = None
            if all(hasattr(x, 'shape') for x in input_list):
                input_shapes = nest.map_structure(lambda x: x.shape, inputs)
            # Only call `build` if the user has manually overridden the build method.
            if not hasattr(self.build, '_is_default'):
                # Any setup work performed only once should happen in an `init_scope`
                # to avoid creating symbolic Tensors that will later pollute any eager
                # operations.
                with tf_utils.maybe_init_scope(self):
                    self.build(input_shapes)
            # We must set self.built since user defined build functions are not
            # constrained to set self.built.
            self.built = True
