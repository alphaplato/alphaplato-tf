#!/bin/python3
#coding:utf-8
#Copyright 2020 Alphaplato. All Rights Reserved.
#Desc:rnn
#=======================================================

from tensorflow.python.util import nest
from tensorflow.python.eager import context
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import dtypes

_concat = rnn_cell_impl._concat

def _transpose_batch_time(x):

  x_static_shape = x.get_shape()
  if x_static_shape.rank is not None and x_static_shape.rank < 2:
    return x

  x_rank = array_ops.rank(x)
  x_t = array_ops.transpose(
      x, array_ops.concat(([1, 0], math_ops.range(2, x_rank)), axis=0))
  x_t.set_shape(
      tensor_shape.TensorShape(
          [x_static_shape.dims[1].value,
           x_static_shape.dims[0].value]).concatenate(x_static_shape[2:]))
  return x_t


def _should_cache():
    """Returns True if a default caching device should be set, otherwise False."""
    if context.executing_eagerly():
        return False
    # Don't set a caching device when running in a loop, since it is possible that
    # train steps could be wrapped in a tf.while_loop. In that scenario caching
    # prevents forward computations in loop iterations from re-reading the
    # updated weights.
    ctxt = ops.get_default_graph()._get_control_flow_context()  # pylint: disable=protected-access
    return control_flow_util.GetContainingWhileContext(ctxt) is None

def dynamic_rnn(cell,
                inputs,
                att_scores=None,
                sequence_length=None,
                initial_state=None,
                dtype=None,
                parallel_iterations=None,
                swap_memory=False,
                time_major=False,
                scope=None):
    with vs.variable_scope(scope or "rnn") as varscope:
        # Create a new scope in which the caching device is either
        # determined by the parent scope, or is set to place the cached
        # Variable using the same placement as for the rest of the RNN.
        if _should_cache():
            if varscope.caching_device is None:
                varscope.set_caching_device(lambda op: op.device)

        # By default, time_major==False and inputs are batch-major: shaped
        #   [batch, time, depth]
        # For internal calculations, we transpose to [time, batch, depth]
        flat_input = nest.flatten(inputs)

        if not time_major:
            # (B,T,D) => (T,B,D)
            flat_input = [ops.convert_to_tensor(
                input_) for input_ in flat_input]
            flat_input = tuple(_transpose_batch_time(input_)
                               for input_ in flat_input)

        parallel_iterations = parallel_iterations or 32
        if sequence_length is not None:
            sequence_length = math_ops.cast(sequence_length, dtypes.int32)
            if sequence_length.get_shape().rank not in (None, 1):
                raise ValueError(
                    "sequence_length must be a vector of length batch_size, "
                    "but saw shape: %s" % sequence_length.get_shape())
            sequence_length = array_ops.identity(  # Just to find it in the graph.
                sequence_length,
                name="sequence_length")

        batch_size = _best_effort_input_batch_size(flat_input)

        if initial_state is not None:
            state = initial_state
        else:
            if not dtype:
                raise ValueError(
                    "If there is no initial_state, you must give a dtype.")
            if getattr(cell, "get_initial_state", None) is not None:
                state = cell.get_initial_state(
                    inputs=None, batch_size=batch_size, dtype=dtype)
            else:
                state = cell.zero_state(batch_size, dtype)

        def _assert_has_shape(x, shape):
            x_shape = array_ops.shape(x)
            packed_shape = array_ops.stack(shape)
            return control_flow_ops.Assert(
                math_ops.reduce_all(math_ops.equal(x_shape, packed_shape)), [
                    "Expected shape for Tensor %s is " % x.name, packed_shape,
                    " but saw shape: ", x_shape
                ])

        if not context.executing_eagerly() and sequence_length is not None:
            # Perform some shape validation
            with ops.control_dependencies(
                    [_assert_has_shape(sequence_length, [batch_size])]):
                sequence_length = array_ops.identity(
                    sequence_length, name="CheckSeqLen")

        inputs = nest.pack_sequence_as(
            structure=inputs, flat_sequence=flat_input)

        (outputs, final_state) = _dynamic_rnn_loop(
            cell,
            inputs,
            state,
            parallel_iterations=parallel_iterations,
            swap_memory=swap_memory,
            att_scores = att_scores,
            sequence_length=sequence_length,
            dtype=dtype)

        # Outputs of _dynamic_rnn_loop are always shaped [time, batch, depth].
        # If we are performing batch-major calculations, transpose output back
        # to shape [batch, time, depth]
        if not time_major:
            # (T,B,D) => (B,T,D)
            outputs = nest.map_structure(_transpose_batch_time, outputs)

        return (outputs, final_state)


def _dynamic_rnn_loop(cell,
                      inputs,
                      initial_state,
                      parallel_iterations,
                      swap_memory,
                      att_scores = None,
                      sequence_length=None,
                      dtype=None):
    state = initial_state
    assert isinstance(parallel_iterations,
                      int), "parallel_iterations must be int"

    state_size = cell.state_size

    flat_input = nest.flatten(inputs)
    flat_output_size = nest.flatten(cell.output_size)

    # Construct an initial output
    input_shape = array_ops.shape(flat_input[0])
    time_steps = input_shape[0]
    batch_size = _best_effort_input_batch_size(flat_input)

    inputs_got_shape = tuple(
        input_.get_shape().with_rank_at_least(3) for input_ in flat_input)

    const_time_steps, const_batch_size = inputs_got_shape[0].as_list()[:2]

    for shape in inputs_got_shape:
        if not shape[2:].is_fully_defined():
            raise ValueError(
                "Input size (depth of inputs) must be accessible via shape inference,"
                " but saw value None.")
        got_time_steps = shape.dims[0].value
        got_batch_size = shape.dims[1].value
        if const_time_steps != got_time_steps:
            raise ValueError(
                "Time steps is not the same for all the elements in the input in a "
                "batch.")
        if const_batch_size != got_batch_size:
            raise ValueError(
                "Batch_size is not the same for all the elements in the input.")

    # Prepare dynamic conditional copying of state & output
    def _create_zero_arrays(size):
        size = _concat(batch_size, size)
        return array_ops.zeros(
            array_ops.stack(size), _infer_state_dtype(dtype, state))

    flat_zero_output = tuple(
        _create_zero_arrays(output) for output in flat_output_size)
    zero_output = nest.pack_sequence_as(
        structure=cell.output_size, flat_sequence=flat_zero_output)

    if sequence_length is not None:
        min_sequence_length = math_ops.reduce_min(sequence_length)
        max_sequence_length = math_ops.reduce_max(sequence_length)
    else:
        max_sequence_length = time_steps

    time = array_ops.constant(0, dtype=dtypes.int32, name="time")

    with ops.name_scope("dynamic_rnn") as scope:
        base_name = scope

    def _create_ta(name, element_shape, dtype):
        return tensor_array_ops.TensorArray(
            dtype=dtype,
            size=time_steps,
            element_shape=element_shape,
            tensor_array_name=base_name + name)

    in_graph_mode = not context.executing_eagerly()
    if in_graph_mode:
        output_ta = tuple(
            _create_ta(
                "output_%d" % i,
                element_shape=(
                    tensor_shape.TensorShape([const_batch_size]).concatenate(
                        _maybe_tensor_shape_from_tensor(out_size))),
                dtype=_infer_state_dtype(dtype, state))
            for i, out_size in enumerate(flat_output_size))
        input_ta = tuple(
            _create_ta(
                "input_%d" % i,
                element_shape=flat_input_i.shape[1:],
                dtype=flat_input_i.dtype)
            for i, flat_input_i in enumerate(flat_input))
        input_ta = tuple(
            ta.unstack(input_) for ta, input_ in zip(input_ta, flat_input))
    else:
        output_ta = tuple([0 for _ in range(time_steps.numpy())]
                          for i in range(len(flat_output_size)))
        input_ta = flat_input

    def _time_step(time, output_ta_t, state, att_scores=None):
        if in_graph_mode:
            input_t = tuple(ta.read(time) for ta in input_ta)
            # Restore some shape information
            for input_, shape in zip(input_t, inputs_got_shape):
                input_.set_shape(shape[1:])
        else:
            input_t = tuple(ta[time.numpy()] for ta in input_ta)

        input_t = nest.pack_sequence_as(
            structure=inputs, flat_sequence=input_t)
        if att_scores is not None:
            # att_score = att_scores[:, time, :]
            att_score = att_scores[:, time]
            call_cell = lambda: cell(input_t, state, att_score)
        else:
            call_cell = lambda: cell(input_t, state)

        if sequence_length is not None:
            (output, new_state) = _rnn_step(
                time=time,
                sequence_length=sequence_length,
                min_sequence_length=min_sequence_length,
                max_sequence_length=max_sequence_length,
                zero_output=zero_output,
                state=state,
                call_cell=call_cell,
                state_size=state_size,
                skip_conditionals=True)
        else:
            (output, new_state) = call_cell()

        output = nest.flatten(output)

        if in_graph_mode:
            output_ta_t = tuple(
                ta.write(time, out) for ta, out in zip(output_ta_t, output))
        else:
            for ta, out in zip(output_ta_t, output):
                ta[time.numpy()] = out
        if att_scores is not None:
            return (time + 1, output_ta_t, new_state, att_scores)
        else:
            return (time + 1, output_ta_t, new_state)

    if att_scores is not None:
        _, output_final_ta, final_state, _ = control_flow_ops.while_loop(
            cond=lambda time, *_: time < time_steps,
            body=_time_step,
            loop_vars=(time, output_ta, state, att_scores),
            parallel_iterations=parallel_iterations,
            swap_memory=swap_memory)
    else:
        _, output_final_ta, final_state = control_flow_ops.while_loop(
            cond=lambda time, *_: time < time_steps,
            body=_time_step,
            loop_vars=(time, output_ta, state),
            parallel_iterations=parallel_iterations,
            swap_memory=swap_memory)

    if in_graph_mode:
        # Make sure that we run at least 1 step, if necessary, to ensure
        # the TensorArrays pick up the dynamic shape.
        loop_bound = math_ops.minimum(time_steps,
                                      math_ops.maximum(1, max_sequence_length))
    else:
        # Using max_sequence_length isn't currently supported in the Eager branch.
        loop_bound = time_steps

    _, output_final_ta, final_state = control_flow_ops.while_loop(
        cond=lambda time, *_: time < loop_bound,
        body=_time_step,
        loop_vars=(time, output_ta, state),
        parallel_iterations=parallel_iterations,
        maximum_iterations=time_steps,
        swap_memory=swap_memory)

    # Unpack final output if not using output tuples.
    if in_graph_mode:
        final_outputs = tuple(ta.stack() for ta in output_final_ta)
        # Restore some shape information
        for output, output_size in zip(final_outputs, flat_output_size):
            shape = _concat([const_time_steps, const_batch_size],
                            output_size,
                            static=True)
            output.set_shape(shape)
    else:
        final_outputs = output_final_ta

    final_outputs = nest.pack_sequence_as(
        structure=cell.output_size, flat_sequence=final_outputs)
    if not in_graph_mode:
        final_outputs = nest.map_structure_up_to(
            cell.output_size, lambda x: array_ops.stack(x, axis=0), final_outputs)

    return (final_outputs, final_state)

def _best_effort_input_batch_size(flat_input):
    for input_ in flat_input:
        shape = input_.shape
        if shape.rank is None:
            continue
        if shape.rank < 2:
            raise ValueError("Expected input tensor %s to have rank at least 2" %
                             input_)
        batch_size = shape.dims[1].value
        if batch_size is not None:
            return batch_size
    # Fallback to the dynamic batch size of the first input.
    return array_ops.shape(flat_input[0])[1]

def _infer_state_dtype(explicit_dtype, state):
    if explicit_dtype is not None:
        return explicit_dtype
    elif nest.is_sequence(state):
        inferred_dtypes = [element.dtype for element in nest.flatten(state)]
        if not inferred_dtypes:
            raise ValueError("Unable to infer dtype from empty state.")
        all_same = all(x == inferred_dtypes[0] for x in inferred_dtypes)
        if not all_same:
            raise ValueError(
                "State has tensors of different inferred_dtypes. Unable to infer a "
                "single representative dtype.")
        return inferred_dtypes[0]
    else:
        return state.dtype

def _maybe_tensor_shape_from_tensor(shape):
    if isinstance(shape, ops.Tensor):
        return tensor_shape.as_shape(tensor_util.constant_value(shape))
    else:
        return shape

def _rnn_step(time,
              sequence_length,
              min_sequence_length,
              max_sequence_length,
              zero_output,
              state,
              call_cell,
              state_size,
              skip_conditionals=False):

    # Convert state to a list for ease of use
    flat_state = nest.flatten(state)
    flat_zero_output = nest.flatten(zero_output)

    # Vector describing which batch entries are finished.
    copy_cond = time >= sequence_length

    def _copy_one_through(output, new_output):
        # TensorArray and scalar get passed through.
        if isinstance(output, tensor_array_ops.TensorArray):
            return new_output
        if output.shape.rank == 0:
            return new_output
        # Otherwise propagate the old or the new value.
        with ops.colocate_with(new_output):
            return array_ops.where(copy_cond, output, new_output)

    def _copy_some_through(flat_new_output, flat_new_state):
        # Use broadcasting select to determine which values should get
        # the previous state & zero output, and which values should get
        # a calculated state & output.
        flat_new_output = [
            _copy_one_through(zero_output, new_output)
            for zero_output, new_output in zip(flat_zero_output, flat_new_output)
        ]
        flat_new_state = [
            _copy_one_through(state, new_state)
            for state, new_state in zip(flat_state, flat_new_state)
        ]
        return flat_new_output + flat_new_state

    def _maybe_copy_some_through():
        """Run RNN step.  Pass through either no or some past state."""
        new_output, new_state = call_cell()

        nest.assert_same_structure(zero_output, new_output)
        nest.assert_same_structure(state, new_state)

        flat_new_state = nest.flatten(new_state)
        flat_new_output = nest.flatten(new_output)
        return control_flow_ops.cond(
            # if t < min_seq_len: calculate and return everything
            time < min_sequence_length,
            lambda: flat_new_output + flat_new_state,
            # else copy some of it through
            lambda: _copy_some_through(flat_new_output, flat_new_state))

    # TODO(ebrevdo): skipping these conditionals may cause a slowdown,
    # but benefits from removing cond() and its gradient.  We should
    # profile with and without this switch here.
    if skip_conditionals:
        # Instead of using conditionals, perform the selective copy at all time
        # steps.  This is faster when max_seq_len is equal to the number of unrolls
        # (which is typical for dynamic_rnn).
        new_output, new_state = call_cell()
        nest.assert_same_structure(zero_output, new_output)
        nest.assert_same_structure(state, new_state)
        new_state = nest.flatten(new_state)
        new_output = nest.flatten(new_output)
        final_output_and_state = _copy_some_through(new_output, new_state)
    else:
        def empty_update(): return flat_zero_output + flat_state
        final_output_and_state = control_flow_ops.cond(
            # if t >= max_seq_len: copy all state through, output zeros
            time >= max_sequence_length,
            empty_update,
            # otherwise calculation is required: copy some or all of it through
            _maybe_copy_some_through)

    if len(final_output_and_state) != len(flat_zero_output) + len(flat_state):
        raise ValueError("Internal error: state and output were not concatenated "
                         "correctly.")
    final_output = final_output_and_state[:len(flat_zero_output)]
    final_state = final_output_and_state[len(flat_zero_output):]

    for output, flat_output in zip(final_output, flat_zero_output):
        output.set_shape(flat_output.get_shape())
    for substate, flat_substate in zip(final_state, flat_state):
        if not isinstance(substate, tensor_array_ops.TensorArray):
            substate.set_shape(flat_substate.get_shape())

    final_output = nest.pack_sequence_as(
        structure=zero_output, flat_sequence=final_output)
    final_state = nest.pack_sequence_as(
        structure=state, flat_sequence=final_state)

    return final_output, final_state