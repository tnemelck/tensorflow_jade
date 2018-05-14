#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 12 06:21:04 2018

@author: elvex
"""

import tensorflow as tf
import input_data_tx as iddc

(train_dir, image_label_adr, 
                image_train_adr, image_validation_adr, image_test_adr,
                label_train_adr, label_validation_adr, label_test_adr,
                IMG_W, IMG_H, NB_CLASSES, LR, log_dir, nb_img_max,
                batch_size, step_iter) = iddc.init_var()

W, H = IMG_W, IMG_H


def cnn_model_fn(features, labels, mode):
      W, H = IMG_W, IMG_H
      """Model function for CNN."""
      # Input Layer
      # Reshape X to 4-D tensor: [batch_size, width, height, channels]
      input_layer = tf.reshape(features["x"], [-1, IMG_W, IMG_H, 3], name = "Reshape")
    
      # Convolutional Layer #1
      # Computes 32 features using a 5x5 filter with ReLU activation.
      # Padding is added to preserve width and height.
      # Input Tensor Shape: [batch_size, W, H, 3]
      # Output Tensor Shape: [batch_size, W, H, 32]
      conv1 = tf.layers.conv2d(
          inputs=input_layer,
          filters=32,
          kernel_size=5,
          strides=1,
          padding="same",
          activation=tf.nn.relu,
          name = "conv_1")
    
      # Pooling Layer #1
      # First max pooling layer with a 2x2 filter and stride of 2
      # Input Tensor Shape: [batch_size, W, H, 32]
      # Output Tensor Shape: [batch_size, W/2, H/2, 32]
      pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=2, strides=2, name = "pool_1")
      W, H = W // 2, H // 2
    
      # Convolutional Layer #2
      # Computes 64 features using a 5x5 filter.
      # Padding is added to preserve width and height.
      # Input Tensor Shape: [batch_size, W, H, 32]
      # Output Tensor Shape: [batch_size, W, H, 64]
      conv2 = tf.layers.conv2d(
          inputs=pool1,
          filters=64,
          kernel_size=5,
          strides=1,
          padding="same",
          activation=tf.nn.relu,
          name = "conv_2")
    
      # Pooling Layer #2
      # Second max pooling layer with a 2x2 filter and stride of 2
      # Input Tensor Shape: [batch_size, W, H, 64]
      # Output Tensor Shape: [batch_size, W/2, H/2, 64]
      pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=2, strides=2, name = "pool_2")
      W, H = W // 2, H // 2
      
      # Convolutional Layer #3
      # Computes 128 features using a 5x5 filter.
      # Padding is added to preserve width and height.
      # Input Tensor Shape: [batch_size, W, H, 64]
      # Output Tensor Shape: [batch_size, W, H, 128]
      conv3 = tf.layers.conv2d(
          inputs=pool2,
          filters=128,
          kernel_size=5,
          strides=1,
          padding="same",
          activation=tf.nn.relu,
          name = "conv_3")
    
      # Pooling Layer #2
      # Second max pooling layer with a 2x2 filter and stride of 2
      # Input Tensor Shape: [batch_size, W, H, 128]
      # Output Tensor Shape: [batch_size, W/2, H/2, 128]
      pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=2, strides=2, name = "pool_3")
      W, H = W // 2, H // 2
    
      # Flatten tensor into a batch of vectors
      # Input Tensor Shape: [batch_size, W, H, 128]
      # Output Tensor Shape: [batch_size, W * H * 128]
      pool2_flat = tf.reshape(pool3, [-1, W * H * 128], name = "flat")
    
      # Dense Layer
      # Densely connected layer with 1024 neurons
      # Input Tensor Shape: [batch_size, 7 * 7 * 64]
      # Output Tensor Shape: [batch_size, 1024]
      dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu, name = "dense_layer")
    
      # Add dropout operation; 0.6 probability that element will be kept
      dropout = tf.layers.dropout(
              inputs=dense, rate=0.7, training=mode == tf.estimator.ModeKeys.TRAIN,
              name = "dropout")
    
      # Logits layer
      # Input Tensor Shape: [batch_size, 1024]
      # Output Tensor Shape: [batch_size, NB_CLASSES]
      logits = tf.layers.dense(inputs=dropout, units=NB_CLASSES, name = "logits")
    
      predictions = {
          # Generate predictions (for PREDICT and EVAL mode)
          "classes": tf.argmax(input=logits, axis=1),
          # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
          # `logging_hook`.
          "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
          }
      if mode == tf.estimator.ModeKeys.PREDICT:
          return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    
      # Calculate Loss (for both TRAIN and EVAL modes)
      loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    
      # Configure the Training Op (for TRAIN mode)
      if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=LR)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
    
      # Add evaluation metrics (for EVAL mode)
      eval_metric_ops = {
          "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
      return tf.estimator.EstimatorSpec(
              mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)