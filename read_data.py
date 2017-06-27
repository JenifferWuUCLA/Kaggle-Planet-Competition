import numpy as np
import os
import re
import random
import math
import pandas as pd
import utils

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
    
    def random_batch(self):
        rand = list()
        for i in xrange(self.batch_size):
            rand.append(self.get_one_random_balance_index()[0])
        batch_imgnames = list()
        for idx in rand:
            batch_imgnames.append(self.imgnames[idx])
        batch_labels = self.labels[rand]

        img_list = list()
        for imgname in batch_imgnames:
            path = self.imgs_path + imgname[0] +".jpg"
            img = utils.load_image(path)
            img_list.append(img)

        batch_imgs = np.reshape(np.stack(img_list), [-1,224,224,3])
        batch_labels = np.reshape(batch_labels, [-1, self.labels.shape[1]])
        return batch_imgs, batch_labels

    def batch(self):
        batch_imgnames = list()
        for idx in range(self.lineidx, self.lineidx+self.batch_size):
            batch_imgnames.append(self.imgnames[idx])
        batch_labels = self.labels[self.lineidx:(self.lineidx+self.batch_size)]
        self.lineidx = self.lineidx+self.batch_size

        img_list = list()
        for imgname in batch_imgnames:
            path = self.imgs_path+imgname+".jpg"
            #print path
            #img = cv2.imread(path)
            #img = cv2.resize(img, (224, 224))
            img = utils.load_image(path)
            img_list.append(img)

        #print batch_labels
        #print img_list
        batch_imgs = np.reshape(np.stack(img_list), [-1,224,224,3])
        batch_labels = np.reshape(batch_labels, [-1, self.labels.shape[1]])
        return batch_imgs, batch_labels
    def read_one(self):
        if len(self.imgnames[self.lineidx]) > 11:
            if self.imgnames[self.lineidx][12] == '-':
                self.lineidx = self.lineidx+1
                return [],[],True
            else:
                img_name = self.imgnames[self.lineidx][:11]
        else:
            img_name = self.imgnames[self.lineidx]
        label = self.labels[self.lineidx]
        self.lineidx = self.lineidx+1

        path = self.imgs_path+img_name+".jpg"
        img = utils.load_image(path)
        #print batch_labels
        #print img_list
        img = np.reshape(img, [-1, 224, 224, 3])
        label = np.reshape(label, [-1, self.labels.shape[1]])
        return img, label, False

