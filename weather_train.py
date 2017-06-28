#train weather label
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import re
import random
import math

import vgg16_trainable as vgg16
import read_data
import utils
import csv

from sklearn.metrics import fbeta_score

def f2_score(y_true, y_pred):
    # fbeta_score throws a confusing error if inputs are not numpy arrays
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # We need to use average='samples' here, any other average method will generate bogus results
    return fbeta_score(y_true, y_pred, beta=2, average='samples')

def get_one_hot_pred(prob):
    y_pred = list()
    for p in prob:
        temp = np.zeros(len(p))
        temp[np.argmax(p)] = 1
        y_pred.append(temp)
    return y_pred
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="2"

class Config():
    batch_size = 8
    steps = "-1"
    gpu = '/gpu:0'

    # checkpoint path and filename
    logdir = "./log"
    params_dir = "./params/"
    load_filename = "vgg16.npy"
    save_filename = params_dir + "vgg16_weather.npy"

    # path
    imgs_path = "./train-jpg/"
    labels_file = "./train_validation_v2_bin.csv"

    # iterations config
    max_iteration = 15000
    summary_iters = 50
    # refer to synset.txt for the order of labels
    # 6: clear, 7: cloudy, 11: haze, 12:partly_cloudy
    usecols = [6, 7, 11, 12]
config = Config()
reader = read_data.Reader(config)

validation_config = Config()
validation_config.labels_file = "./validation_train_v2_bin.csv"
validation_reader = read_data.Reader(validation_config)

images = tf.placeholder(tf.float32, [config.batch_size, 224, 224, 3])
true_out = tf.placeholder(tf.float32, [config.batch_size, len(config.usecols)])
train_mode = tf.placeholder(tf.bool)

vgg = vgg16.Vgg16(config.load_filename, output_size=4)
vgg.build(images, train_mode)
print vgg.get_var_count() , "variables"
with tf.name_scope('loss'):
    cost = tf.reduce_mean(-tf.reduce_sum(true_out * tf.log(vgg.prob), [1]))
    tf.summary.scalar('loss', cost)
    
with tf.name_scope('train'):
    rate = 1e-3
    train = tf.train.GradientDescentOptimizer(rate).minimize(cost)
    tf.summary.scalar('learning_rate', rate)
    tf.summary.scalar('batch_size', config.batch_size)
    
    merged = tf.summary.merge_all()

with tf.device(config.gpu):    
    sess = tf.Session()
    writer = tf.summary.FileWriter(config.logdir, sess.graph)
    sess.run(tf.global_variables_initializer())

    print "start training"
    # start training
    for idx in xrange(config.max_iteration):
        imgs, labels = reader.random_batch()
        # feed data into the model
        feed_dict = {
            images : imgs,
            true_out : labels,
            train_mode : True
        }
        sess.run(train, feed_dict=feed_dict)
        if  idx % 50 == 0:
            result = sess.run(merged, feed_dict=feed_dict)
            loss = sess.run(cost, feed_dict=feed_dict)
            
            # f2 score
            prob = sess.run(vgg.prob, feed_dict=feed_dict)
            y_pred = get_one_hot_pred(prob)
            f2 = f2_score(labels, y_pred)
            
            print idx, "cost:", loss, "f2_score:", f2
            writer.add_summary(result, idx)
            if idx % 1000 == 0:
                valid_pred = []
                valid_true_out = []
                for x in  xrange(np.int32(np.ceil(4048/config.batch_size))):
                    valid_img, valid_label = validation_reader.batch()
                    valid_feed_dict = {
                        images : valid_img,
                        true_out: valid_label,
                        train_mode : False
                    }
                    valid_prob = sess.run(vgg.prob, feed_dict=valid_feed_dict)
                    valid_pred = np.append(valid_pred, get_one_hot_pred(valid_prob))
                    valid_true_out = np.append(valid_true_out, valid_label)
                valid_pred = np.reshape(valid_pred,[-1, len(config.usecols)])
                valid_true_out = np.reshape(valid_true_out, [-1, len(config.usecols)])
                valid_f2 = f2_score(valid_true_out, valid_pred)
                print "validation_f2_score:", valid_f2
                vgg.save_npy(sess, config.save_filename)
