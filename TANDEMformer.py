'''
Tensorflow implementation of the TANDEMformer 
equipped with Normalized Summation Pooling (NSP) layer.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, BatchNormalization, Activation, LayerNormalization
from tensorflow.keras import layers 
import pdb


class TANDEMformer(tf.keras.Model):
    # pools over duration dimension to avoid zero-padding

    def __init__(self, nb_cls, duration, feat_dim, 
                num_transformer_blocks, head_size, num_heads, dropout,
                ff_dim, mlp_units,
                ifNorm=True, ifPE=True, activation='relu', mlp_activation='relu',
                alpha=1.0):
        
        super(TANDEMformer, self).__init__(name="TANDEMformer")

        # custom activation functions
        def B2Bsqrt(x):
            return tf.math.sign(x) * (
                tf.math.sqrt(alpha + tf.math.abs(x)) - tf.math.sqrt(alpha)
        )
        def B2Blog(x):
            return tf.math.sign(x) * (
                tf.math.log(1 + tf.math.abs(x))
        )
        def combHsine(x):
            return tf.math.sinh(alpha * x) + tf.math.asinh(alpha * x)
        
        if activation.lower() == 'b2bsqrt':
            self.activation = B2Bsqrt
        elif activation.lower() == 'b2blog':
            self.activation = B2Blog
        elif activation.lower() == 'combhsine':
            self.activation = combHsine
        else:
            self.activation = activation
        
        if mlp_activation.lower() == 'b2bsqrt':
            self.mlp_activation = B2Bsqrt
        elif mlp_activation.lower() == 'b2blog':
            self.mlp_activation = B2Blog
        elif mlp_activation.lower() == 'combhsine':
            self.mlp_activation = combHsine
        else:
            self.mlp_activation = mlp_activation

        # transformer building blocks
        self.MultiHeadAttention = layers.MultiHeadAttention(
                key_dim=head_size, num_heads=num_heads, dropout=dropout)
        self.conv1 = layers.Conv1D(
            filters=ff_dim, kernel_size=1, activation=self.activation)
        self.conv2 = layers.Conv1D(
            filters=feat_dim, kernel_size=1, activation=None)
        self.LayerNorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.LayerNorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.Dropout = layers.Dropout(dropout)
        
        # Parameters
        self.nb_cls = nb_cls
        self.num_transformer_blocks = num_transformer_blocks
        self.ifNorm = ifNorm
        self.ifPE = ifPE # positional embedding

        # logit computation
        self.Dense1 = layers.Dense(mlp_units, activation=self.mlp_activation, use_bias=True)
        self.Dense2 = layers.Dense(nb_cls, activation=None, use_bias=True)

    # transformer block 
    def transformer_encoder(self, inputs, training):
        # self-attention 
        x = inputs # (batch_size, duration, feat_dim)
        if self.ifNorm:
            x = self.LayerNorm1(x, training=training)
        x = self.MultiHeadAttention(query=x, value=x, training=training)
        x = self.Dropout(x, training=training)
        res = x + inputs

        # feedforward
        if self.ifNorm:
            x = self.LayerNorm2(res, training=training)
        else:
            x = res
        x = self.conv1(x, training=training)
        x = self.Dropout(x, training=training)
        x = self.conv2(x, training=training)
        return x + res

    def transformer_classifier(self, inputs, training, max_duration):
        x = inputs # (batch_size, duration, feat_dim)
        for _ in range(self.num_transformer_blocks):
            x = self.transformer_encoder(x, training=training) 
        
        ##### NSP #####
        x = tf.reduce_sum(x, axis=1) / max_duration
        #### NSP end ###
        x = self.Dense1(x, training=training)
        x = self.Dropout(x, training=training)
        outputs = self.Dense2(x)
        return outputs
    
    def PositionalEncoding(self, inputs, max_freq=10000):
        assert len(inputs.shape) == 3 # input size: (batch_size, duration, feat_dim)
        batch_size = inputs.shape[0]
        maxpos = inputs.shape[-2]
        d_model = inputs.shape[-1]
        x = np.arange(0, maxpos)
        y = np.arange(0, d_model)
        xx, yy = np.meshgrid(x, y, indexing='ij')
        arg = xx/max_freq**(2*(yy//2)/d_model)
        cos_mask = y % 2
        sin_mask = 1 - cos_mask
        PE = np.sin(arg) * sin_mask + tf.cos(arg) * cos_mask
        return np.repeat(PE[np.newaxis, :, :], batch_size, axis=0)

    def call(self, inputs, training):
        """Calc logits.
        Here, "duration" can be smaller than the original length T, depending on the Markov order N
        Args:
            inputs: A Tensor with shape=(batch, duration, feature dimension). E.g. (83, 50, 512) for SiW.
            training: A boolean. Training flag used in BatchNormalization and dropout.
        Returns:
            A Tensor with shape=(batch, duration, nb_cls).
        """
        duration = inputs.shape[1]
        outputs_pool = []
        for i in range(duration):
            targinputs = inputs[:, :i+1, :]
            if self.ifPE: # positional encoding
                targinputs += self.PositionalEncoding(targinputs)
            outputs_pool.append(
                self.transformer_classifier(
                    targinputs, 
                training=training, max_duration=duration)
            )
            
        return tf.stack(outputs_pool, axis=1) # A Tensor with shape=(batch, effective duration, nb_cls)
