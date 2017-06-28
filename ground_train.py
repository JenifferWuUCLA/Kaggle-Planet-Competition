land_labels = ['agriculture', 'artisinal_mine', 'bare_ground', 'blooming', 'blow_down', 
               'conventional_mine', 'cultivation', 'habitation','primary', 'road',
               'selective_logging', 'slash_burn', 'water']

#train ground label with validation set
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
from sklearn.metrics import accuracy_score

def f2_score(y_true, y_pred):
    # fbeta_score throws a confusing error if inputs are not numpy arrays
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # We need to use average='samples' here, any other average method will generate bogus results
    return fbeta_score(y_true, y_pred, beta=2, average='samples')

def get_one_hot_by_thres(prob):
    thres = [0.07, 0.17, 0.2, 0.04, 0.23, 0.22, 0.1, 0.19, 0.12, 0.14, 0.25, 0.26, 0.16]
    y_pred = list()
    array_len = len(prob[0])
    for p in prob:
        temp = np.zeros(array_len)
        for i in xrange(array_len):
            if p[i] > thres[i]:
                temp[i] = 1
        y_pred.append(temp)
    return y_pred

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"

class Config():
    batch_size = 8
    steps = "-1"
    gpu = '/gpu:0'

    # checkpoint path and filename
    logdir = "./log"
    params_dir = "./params/"
    load_filename = "vgg16.npy"
    save_filename = params_dir + "vgg16_ground.npy"

    # path
    imgs_path = "./train-jpg/"
    labels_file = "./train_validation_v2_bin.csv"

    # iterations config
    max_iteration = 10000
    summary_iters = 50
    # refer to synset.txt for the order of labels
    # ground labels
    usecols = [1, 2, 3, 4, 5, 8, 9, 10, 13, 14, 15, 16, 17]

config = Config()
reader = read_data.Reader(config)

validation_config = Config()
validation_config.labels_file = "./validation_train_v2_bin.csv"
validation_reader = read_data.Reader(validation_config)

images = tf.placeholder(tf.float32, [None, 224, 224, 3])
true_out = tf.placeholder(tf.float32, [None, len(config.usecols)])
train_mode = tf.placeholder(tf.bool)

vgg = vgg16.Vgg16(config.load_filename, output_size=len(config.usecols))
vgg.build(images, train_mode)
print vgg.get_var_count() , "variables"
with tf.name_scope('loss'):
    cost = tf.reduce_mean(-tf.reduce_sum(true_out * tf.log(vgg.prob), [1]))
    tf.summary.scalar('loss', cost)
    valid_f2_score = 0
    tf.summary.scalar('validf2_score', valid_f2_score)
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
                        
            print idx, "cost:", loss
            writer.add_summary(result, idx)
            if idx % 500 == 0:
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
                    valid_pred = np.append(valid_pred, get_one_hot_by_thres(valid_prob))
                    valid_true_out = np.append(valid_true_out, valid_label)
                valid_pred = np.reshape(valid_pred,[-1, len(config.usecols)])
                valid_true_out = np.reshape(valid_true_out, [-1, len(config.usecols)])
                valid_f2_score = f2_score(valid_true_out, valid_pred)
                print "validation_f2_score:", valid_f2_score
                
                for i in xrange(13):
                    acy_score = accuracy_score(np.transpose(valid_true_out)[i], np.transpose(valid_pred)[i])
                    print land_labels[i], "acy_score:\t", acy_score
                vgg.save_npy(sess, config.save_filename)
