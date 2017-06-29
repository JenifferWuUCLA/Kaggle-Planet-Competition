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

class Config():
    batch_size = 32
    steps = "-1"
    gpu = '/gpu:0'

    # checkpoint path and filename
    logdir = "./log/train_log/"
    params_dir = "./params/"
    load_filename = params_dir + "vgg16_all_2.npy"
    save_filename = params_dir + "vgg16.npy"

    # path
    imgs_path = "./test-jpg/"
    labels_file = "./sample_submission_v2.csv"
    usecols = range(1,18)
    # test number
    #number = 91
    number = 61191

config = Config()

with tf.device(config.gpu):
    sess = tf.Session()
    images = tf.placeholder(tf.float32, [None, 224, 224, 3])
    train_mode = tf.placeholder(tf.bool)

    vgg = vgg16.Vgg16(config.load_filename)
    vgg.build(images, train_mode)

    # print number of variables used: 143667240 variables, i.e. ideal size = 548MB
    print(vgg.get_var_count())
    
    sess.run(tf.global_variables_initializer())
    
    imgnames = pd.read_csv(config.labels_file, skiprows=[0], usecols=[0], header=None).values
    thres = [0.07, 0.01, 0.07, 0.05, 0.05, 0.06, 0.48, 0.01, 0.05, 0.05, 0.09, 0.04, 0.08, 0.08, 0.04, 0.02, 0.06]
    
    synset = [l.strip() for l in open("./synset.txt").readlines()]
    
    pred = []
    true_out = []
    lineidx = 0
    
    for x in  xrange(np.int32(np.ceil(config.number/float(config.batch_size)))):
        batch_imgnames = list()
        lineidx_upper = lineidx + config.batch_size
        if lineidx_upper > config.number:
            lineidx_upper = config.number
        for idx in range(lineidx, lineidx_upper):
            batch_imgnames.append(imgnames[idx])
        lineidx = lineidx_upper

        img_list = list()
        for imgname in batch_imgnames:
            path = config.imgs_path + imgname[0] + ".jpg"
            img = utils.load_image(path)
            img_list.append(img)
        
        img = np.reshape(np.stack(img_list), [-1,224,224,3])
        
        feed_dict = {
            images : img,
            train_mode : False
        }
        prob = sess.run(vgg.prob, feed_dict=feed_dict)
        pred = np.append(pred, prob)
        print x, "times"
    pred = np.reshape(pred,[-1, len(config.usecols)])
    pred_out = np.zeros_like(pred)
    
    for i in range(len(config.usecols)):
        pred_out[:, i] = (pred[:, i] > thres[i]).astype(np.int)
    
    print "writing data"
    # writting data
    with open('./result.csv', 'wb') as f:
        writer = csv.writer(f)
        writer.writerow(['image_name', 'tags'])
        for i in xrange(config.number):
            tag = ""
            for j in xrange(len(config.usecols)):
                if pred_out[i,j] == 1:
                    tag = tag + synset[j] + ' '
                    #if j == 6:
                        #break
            row = [imgnames[i][0]] + [tag[:-1]]
            writer.writerow(row)
    
