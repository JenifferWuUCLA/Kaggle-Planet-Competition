import numpy as np
import os
import re
import random
import math
import pandas as pd
import utils
import cv2
from keras.preprocessing import image
from inception_v3 import preprocess_input

class Reader():

    def __init__(self, config):
        
        self.imgnames = list()
        self.labels = pd.read_csv(config.labels_file, skiprows=[0], usecols=config.usecols, header=None)
        self.labels = np.float32(self.labels.values)

        self.imgnames = pd.read_csv(config.labels_file, skiprows=[0], usecols=[0], header=None).values

        self.size = len(self.imgnames)
        self.batch_size = config.batch_size
        self.imgs_path = config.imgs_path
        self.lineidx = 0
        self.sample_num = len(self.imgnames)
        self.lable_tpyes_num = len(config.usecols)
        self.current_sample_index = 0
        self.current_sample = list()
        for i in xrange(self.lable_tpyes_num):
            self.current_sample.append([index for index, value in enumerate(np.transpose(self.labels)[i]) if value == 0])
    # this method can return the index of every type of sample one by one when fetch random batch
    def get_one_random_balance_index(self):
        rand_index = random.sample(self.current_sample[self.current_sample_index], 1)
        self.current_sample_index = (self.current_sample_index + 1) % self.lable_tpyes_num 
        return rand_index
    
    def random_batch(self, case):
        rand = list()
        for i in xrange(self.batch_size):
            rand.append(self.get_one_random_balance_index()[0])
        batch_imgnames = list()
        for idx in rand:
            batch_imgnames.append(self.imgnames[idx])
        batch_labels = self.labels[rand]

        img_list = list()
        for imgname in batch_imgnames:
            img_path = self.imgs_path + imgname[0] +".jpg"
            img = image.load_img(img_path, target_size=(299,299))
            x = image.img_to_array(img)
            if case != 0:
                x = image_preprocess(x, i)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            img_list.append(x)
        
        batch_imgs = np.reshape(np.stack(img_list), [-1,299,299,3])
        batch_labels = np.reshape(batch_labels, [-1, self.labels.shape[1]])
        return batch_imgs, batch_labels

    def batch(self, case):
        batch_imgnames = list()
        lineidx_upper = self.lineidx + self.batch_size
        if lineidx_upper > self.sample_num:
            lineidx_upper = self.sample_num
        for idx in range(self.lineidx, lineidx_upper):
            batch_imgnames.append(self.imgnames[idx])
        batch_labels = self.labels[self.lineidx:lineidx_upper]
        self.lineidx = lineidx_upper

        if self.lineidx >= self.sample_num:
            self.lineidx = 0

        img_list = list()
        for imgname in batch_imgnames:
            img_path = self.imgs_path + imgname[0] + ".jpg"
            img_path = self.imgs_path + imgname[0] +".jpg"
            img = image.load_img(img_path, target_size=(299, 299))
            x = image.img_to_array(img)
            if case > 0:
                x = image_preprocess(x, case)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            img_list.append(x)
        
        batch_imgs = np.reshape(np.stack(img_list), [-1,299,299,3])
        batch_labels = np.reshape(batch_labels, [-1, self.labels.shape[1]])
        return batch_imgs, batch_labels

def image_preprocess(x, case):
    rows = x.shape[0]
    cols = x.shape[1]
    if case == 1:
        M = cv2.getRotationMatrix2D((rows/2,cols/2), 90, 1)
        x = cv2.warpAffine(x,M,(rows,cols))
    elif case == 2:
        M = cv2.getRotationMatrix2D((rows/2,cols/2), 180, 1)
        x = cv2.warpAffine(x,M,(rows,cols))
    elif case == 3:
        M = cv2.getRotationMatrix2D((rows/2,cols/2), 270, 1)
        x = cv2.warpAffine(x,M,(rows,cols))
    elif case == 4:
        x = np.fliplr(x)
    elif case == 5:
        x = np.flipud(x)
    elif case == 6:
        for i in range(3):
            x[:,:,i] = np.transpose(x[:,:,i])
    return x

def random_batch_generator(config):
    reader = Reader(config)
    while True:
        batch_features, batch_labels = reader.random_batch(config.case)
        yield batch_features, batch_labels
def batch_generator(config):
    reader = Reader(config)
    while True:
        batch_features, batch_labels = reader.batch(config.case)
        yield batch_features, batch_labels
