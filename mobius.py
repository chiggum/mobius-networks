# -*- coding: utf-8 -*-
import numpy as np

import copy
import types as python_types
import warnings

from keras import backend as K
from keras import activations
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.engine import InputSpec
from keras.engine import Layer
from keras.utils.generic_utils import func_dump
from keras.utils.generic_utils import func_load
from keras.utils.generic_utils import deserialize_keras_object
from keras.utils.generic_utils import has_arg

class Mobius(Layer):
    """
    Note: if the input to the layer has a rank greater than 2, then
    it is flattened prior to the initial dot product with `kernel`.

    # Arguments
        units: Positive integer, dimensionality of the output space.
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).

    # Input shape
        nD tensor with shape: `(batch_size, ..., input_dim)`.
        The most common situation would be
        a 2D input with shape `(batch_size, input_dim)`.

    # Output shape
        nD tensor with shape: `(batch_size, ..., units)`.
        For instance, for a 2D input with shape `(batch_size, input_dim)`,
        the output would have shape `(batch_size, units)`.
    """
    def __init__(self, units,
                 activation=None,
                 use_bias_1=True,
                 use_bias_2=True,
                 kernel_initializer='orthogonal',
                 bias_1_initializer='zeros',
                 bias_2_initializer='zeros',
                 scaler_initializer='ones',
                 kernel_regularizer=None,
                 bias_1_regularizer=None,
                 bias_2_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_1_constraint=None,
                 bias_2_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(Mobius, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias_1 = use_bias_1
        self.use_bias_2 = use_bias_2
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_1_initializer = initializers.get(bias_1_initializer)
        self.bias_2_initializer = initializers.get(bias_2_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_1_regularizer = regularizers.get(bias_1_regularizer)
        self.bias_2_regularizer = regularizers.get(bias_2_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_1_constraint = constraints.get(bias_1_constraint)
        self.bias_2_constraint = constraints.get(bias_2_constraint)
        self.scaler_initializer = scaler_initializer
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]
        self.input_dim = input_dim

        self.kernel = self.add_weight(shape=(self.input_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias_1:
            self.bias_1 = self.add_weight(shape=(self.input_dim,),
                                        initializer=self.bias_1_initializer,
                                        name='bias_1',
                                        regularizer=self.bias_1_regularizer,
                                        constraint=self.bias_1_constraint)
        else:
            self.bias_1 = None
        
        if self.use_bias_2:
            self.bias_2 = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_2_initializer,
                                        name='bias_2',
                                        regularizer=self.bias_2_regularizer,
                                        constraint=self.bias_2_constraint)
        else:
            self.bias_2 = None
        
        self.scaler = self.add_weight(shape=(1,),
                                        initializer=self.scaler_initializer,
                                        name="scaler",
                                        regularizer=None,
                                        constraint=None)

        self.input_spec = InputSpec(min_ndim=2, axes={-1: self.input_dim})
        self.built = True

    def call(self, inputs):
        if self.use_bias_1:
            divider = K.expand_dims(K.sum(K.square(K.bias_add(inputs, -self.bias_1)), 1))
            output = self.scaler*K.dot(K.bias_add(inputs, -self.bias_1)/divider, self.kernel)
        else:
            divider = K.expand_dims(K.sum(K.square(inputs), 1))
            output = self.scaler*K.dot(inputs/divider, self.kernel)
        
        if self.use_bias_2:
            output = K.bias_add(output, self.bias_2)
        if self.activation is not None:
            output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def get_config(self):
        config = {
            'units': self.units,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_1_initializer': initializers.serialize(self.bias_1_initializer),
            'bias_2_initializer': initializers.serialize(self.bias_2_initializer),
            'scaler_initializer': initializers.serialize(self.scaler_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_1_regularizer': regularizers.serialize(self.bias_1_regularizer),
            'bias_2_regularizer': regularizers.serialize(self.bias_2_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_1_constraint': constraints.serialize(self.bias_1_constraint),
            'bias_2_constraint': constraints.serialize(self.bias_2_constraint)
        }
        base_config = super(Mobius, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))