#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 12 07:23:24 2018

@author: elvex
"""

import tensorflow as tf
import input_data_tx as iddc
from model import cnn_model_fn

(train_dir, image_label_adr, 
                image_train_adr, image_validation_adr, image_test_adr,
                label_train_adr, label_validation_adr, label_test_adr,
                IMG_W, IMG_H, NB_CLASSES, LR, log_dir, nb_img_max,
                batch_size, step_iter) = iddc.init_var()


def serving_input_receiver_fn():
    serialized_tf_example = tf.placeholder(dtype=tf.string, shape=[None], name='input_tensors')
    receiver_tensors      = {"predictor_inputs": serialized_tf_example}
    feature_spec          = {"words": tf.FixedLenFeature([25],tf.int64)}
    features              = tf.parse_example(serialized_tf_example, feature_spec)
    return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)


def train(adr):
  tf.reset_default_graph()
  # Load training and eval data
  train_data, train_labels = iddc.get_tensor_train()
  
  
  with tf.Session() as s:
      (data, labels) = s.run([train_data, train_labels])

  # Create the Estimator
  classifier = tf.estimator.Estimator(
      model_fn=cnn_model_fn, model_dir=log_dir)

  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=10)

  # Train the model
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": data},
      y=labels,
      batch_size=batch_size,
      num_epochs=None,
      shuffle=True)
  classifier.train(
      input_fn=train_input_fn,
      steps=step_iter,
      hooks=[logging_hook])
  
  #Save th model
  full_model_dir = classifier.export_savedmodel(export_dir_base=adr, serving_input_receiver_fn=serving_input_receiver_fn)
  return None
  
def test(adr):
  test_data, test_labels = iddc.get_tensor_test()
  with tf.Session() as s:
      (data, labels) = s.run([test_data, test_labels])
      tf.saved_model.loader.load(s, [tf.saved_model.tag_constants.SERVING], adr)
      classifier = tf.contrib.predictor.from_saved_model(adr)

  # Evaluate the model and print results
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": data},
      y=labels,
      num_epochs=1,
      shuffle=False)
  eval_results = classifier.evaluate(input_fn=eval_input_fn)
  print(eval_results)
