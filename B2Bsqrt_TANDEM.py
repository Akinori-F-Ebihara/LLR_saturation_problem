"""
Tensorflow implementation of the B2Bsqrt-TANDEM
equipped with the B2B-sqrt activation function.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, BatchNormalization, Activation, LayerNormalization
from tensorflow.keras import layers

class B2Bsqrt_TANDEM(tf.keras.Model):
    # B2B: custom activation function with range [-Inf, Inf]

    def __init__(self, nb_cls, width_lstm, duration, dropout=0., 
                 activation='relu', recurrent_activation='relu', fc_activation='relu',
                 ifBN=True, alpha=0.01):
        
        super(B2Bsqrt_TANDEM, self).__init__(name="B2Bsqrt_TANDEM")

        # custom activation functions
        def B2Bsqrt(x):
            return tf.math.sign(x) * (
                tf.math.sqrt(alpha + tf.math.abs(x)) - tf.math.sqrt(alpha)
        )
        def B2Blog(x):
            return tf.math.sign(x) * (
                tf.math.log(1 + tf.math.abs(x))
        )
        def customReLU(x):
            return tf.math.maximum(0, x)

        # Parameters
        self.nb_cls = nb_cls
        self.width_lstm = width_lstm
        self.dropout = dropout

        if activation.lower() == 'b2bsqrt':
            self.activation = B2Bsqrt
        elif activation.lower() == 'b2blog':
            self.activation = B2Blog
        elif activation.lower() == 'customrelu':
            self.activation = customReLU
        else:
            self.activation = activation

        if recurrent_activation.lower() == 'b2bsqrt':
            self.recurrent_activation = B2Bsqrt
        elif recurrent_activation.lower() == 'b2blog':
            self.recurrent_activation = B2Blog
        elif recurrent_activation.lower() == 'customrelu':
            self.recurrent_activation = customReLU
        else:
            self.recurrent_activation = recurrent_activation
        self.fc_activation = fc_activation
        self.ifBN = ifBN

        # RNN
        self.rnn = tf.keras.layers.LSTM(
            units=self.width_lstm, 
            activation=Activation(self.activation), 
            recurrent_activation=Activation(self.recurrent_activation),
            use_bias=True, 
            return_sequences=True, return_state=True
        )
        
        self.LN = LayerNormalization(epsilon=1e-6)
        self.activation_logit = Activation(self.fc_activation)
        self.FC = Dense(nb_cls, activation=None, use_bias=True)
    
    def TDBN_act_fc_logit(self, x, training, duration):
        """
        Args:
            x: A Tensor. Output of LSTM with shape=(batch, duration, self.width_lstm).
        Return:
            x: A Tensor. Logit with shape=(batch, duration, self.nb_cls)        
        """
        
        # apply LayerNormalization
        # if self.ifBN:
        #     x = self.LN(x, training=training)
        y_stack = []
        for i in range(duration):
            y = x[:, i, :]
            if self.ifBN:
                y = self.LN(y, training=training)
            y = self.activation_logit(y)
            y = self.FC(y)
            y_stack.append(y)
        y_stack = tf.stack(y_stack, axis=1)
        
        return y_stack

    def call(self, inputs, training):
        """Calc logits.
        Args:
            inputs: A Tensor with shape=(batch, duration, feature dimension). E.g. (128, 20, 784) for nosaic MNIST.
            training: A boolean. Training flag used in BatchNormalization and dropout.
        Returns:
            outputs: A Tensor with shape=(batch, duration, nb_cls).
        """
        # Parameters
        inputs_shape = inputs.shape 
        duration = inputs_shape[1] # 20 by default for nosaic MNIST

        # Feedforward
        outputs, _, _ = self.rnn(inputs, training=training)

        # Make logits
#         outputs = tf.reshape(outputs, (-1, self.width_lstm))
        outputs = self.TDBN_act_fc_logit(
            outputs, training=training, duration=duration)
#         outputs = tf.reshape(outputs, (-1, duration, self.nb_cls)) # (B, T, nb_cls)

        return outputs # A Tensor with shape=(batch, duration, nb_cls)
