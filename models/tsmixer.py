# coding=utf-8
"""Implementation of TSMixer (fixed for Keras compatibility)."""

import tensorflow as tf
from tensorflow.keras import layers


def res_block(inputs, norm_type, activation, dropout, ff_dim):
  """Residual block of TSMixer."""

  norm = (
      layers.LayerNormalization
      if norm_type == 'L'
      else layers.BatchNormalization
  )

  # Temporal Linear
  # x = norm(axis=[-2, -1])(inputs)

  # بدل tf.transpose -> layers.Permute
  x = layers.Permute((2, 1))(inputs)  # [Batch, Channel, Input Length]
  x = layers.Dense(tf.shape(inputs)[1], activation=activation)(x)
  x = layers.Permute((2, 1))(x)       # [Batch, Input Length, Channel]
  x = layers.Dropout(dropout)(x)
  res = x + inputs

  # Feature Linear
  # x = norm(axis=[-2, -1])(res)
  x = layers.Dense(ff_dim, activation=activation)(res)
  x = layers.Dropout(dropout)(x)
  x = layers.Dense(inputs.shape[-1])(x)
  x = layers.Dropout(dropout)(x)
  return x + res


def build_model(
    input_shape,
    pred_len,
    norm_type,
    activation,
    n_block,
    dropout,
    ff_dim,
    target_slice,
):
  """Build TSMixer model."""

  inputs = tf.keras.Input(shape=input_shape)
  x = inputs  # [Batch, Input Length, Channel]
  for _ in range(n_block):
    x = res_block(x, norm_type, activation, dropout, ff_dim)

  if target_slice:
    x = x[:, :, target_slice]

  # بدل tf.transpose -> Permute
  x = layers.Permute((2, 1))(x)       # [Batch, Channel, Input Length]
  x = layers.Dense(pred_len)(x)       # [Batch, Channel, Output Length]
  outputs = layers.Permute((2, 1))(x) # [Batch, Output Length, Channel]
  return tf.keras.Model(inputs, outputs)
