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

# The arguments y and p are both Nx17 numpy array arrays
# where N is the number of rows in the validation set.
# y is true_label, p is the train result

def optimise_f2_thresholds(y, p, thres, isAll, verbose=True, resolution=100):
    
    type_num = y.shape[1]
    def mf(x):
        p2 = np.zeros_like(p)
        for i in range(type_num):
            p2[:,i] = (p[:,i] > x[i]).astype(np.int)
            # consider relation between classes in all classes
            if isAll:
                for i in range(p2.shape[0]):
                    # if cloudy others are all zero
                    if p2[i,6] == 1:
                        p2[i,:] = np.zeros(p2.shape[1]).astype(np.int)
                        p2[i,6] = 1
                    # if slash_burn, blooming, selective_logging, primary are 1
                    #if (p2[i,3] + p2[i,14] + p2[i,15]) > 0:
                    #    p2[i,13] = 1
        score = fbeta_score(y, p2, beta=2, average='samples')
        return score

    x = thres
    for i in range(type_num):
        best_i2 = 0
        best_score = 0
        for i2 in range(resolution):
            i2 = i2 / float(resolution)
            x[i] = i2
            score = mf(x)
            if score > best_score:
                best_i2 = i2
                best_score = score
        x[i] = best_i2
        if verbose:
            print(i, best_i2, best_score)
    return x

def f2_score(y_true, y_pred):
    # fbeta_score throws a confusing error if inputs are not numpy arrays
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # We need to use average='samples' here, any other average method will generate bogus results
    return fbeta_score(y_true, y_pred, beta=2, average='samples')

def g_train(__config, valid_config, threshold):
    config = __config
    labels = ['tags','agriculture', 'artisinal_mine', 'bare_ground', 'blooming', 'blow_down', 'clear',
              'cloudy', 'conventional_mine', 'cultivation', 'habitation', 'haze', 'partly_cloudy',
              'primary', 'road', 'selective_logging', 'slash_burn', 'water']
    land_labels = np.array(labels)[config.usecols].tolist()
    
    reader = read_data.Reader(config)
    validation_config = valid_config
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
        #writer = tf.summary.FileWriter(config.logdir, sess.graph)
        sess.run(tf.global_variables_initializer())

        print "start training"
        # start training
        # init thres
        thres = threshold

        for idx in xrange(config.max_iteration):
            imgs, labels = reader.random_batch()
            # feed data into the model
            feed_dict = {
                images : imgs,
                true_out : labels,
                train_mode : True
            }
            sess.run(train, feed_dict=feed_dict)
            if  idx % config.summary_iters == 0:
                result = sess.run(merged, feed_dict=feed_dict)
                loss = sess.run(cost, feed_dict=feed_dict)

                print idx, "cost:", loss
                #writer.add_summary(result, idx)
                if idx % config.valid_iters == 0:
                    valid_pred = []
                    valid_true_out = []
                    for x in  xrange(np.int32(np.ceil(4048/float(config.batch_size)))):
                        valid_img, valid_label = validation_reader.batch()
                        valid_feed_dict = {
                            images : valid_img,
                            true_out: valid_label,
                            train_mode : False
                        }
                        valid_prob = sess.run(vgg.prob, feed_dict=valid_feed_dict)
                        valid_pred = np.append(valid_pred, valid_prob)
                        valid_true_out = np.append(valid_true_out, valid_label)
                    valid_pred = np.reshape(valid_pred,[-1, len(config.usecols)])
                    valid_true_out = np.reshape(valid_true_out, [-1, len(config.usecols)])

                    valid_pred_out = np.zeros_like(valid_pred)
                    for i in range(len(config.usecols)):
                        valid_pred_out[:, i] = (valid_pred[:, i] > thres[i]).astype(np.int)

                    valid_f2_score = f2_score(valid_true_out, valid_pred_out)

                    print "validation_f2_score:", valid_f2_score

                    for i in xrange(len(config.usecols)):
                        acy_score = accuracy_score(valid_true_out[:, i], valid_pred_out[:, i])
                        print "acy_score:\t", land_labels[i], "\t", acy_score
                        
                    thres = optimise_f2_thresholds(np.array(valid_true_out), np.array(valid_pred),
                                                   thres, len(config.usecols) == 17, verbose=True, resolution=100)
                    print "best thres:", thres

                    vgg.save_npy(sess, config.save_filename)
