#predict
import numpy as np
import csv
import pandas as pd
from fbeta_score import f2_score, optimise_f2_thresholds
from read_data_extend import Reader, preprocess_input, image_preprocess
from keras.models import load_model
from keras.preprocessing import image

class read_img():
    def __init__(self, config):
        self.imgnames = pd.read_csv(config.labels_file, skiprows=[0], usecols=[0], header=None).values
        self.lineidx = 0
        self.config = config
    def reset(self):
        self.lineidx = 0
    def read_test_img(self, case):
        batch_imgnames = list()
        lineidx_upper = self.lineidx + self.config.batch_size
        if lineidx_upper > self.config.number:
            lineidx_upper = self.config.number
        for idx in range(self.lineidx, lineidx_upper):
            batch_imgnames.append(self.imgnames[idx])
        self.lineidx = lineidx_upper

        img_list = list()
        for imgname in batch_imgnames:
            img_path = config.imgs_path + imgname[0] + ".jpg"
            x = image.load_img(img_path, target_size=(299, 299))
            x = image.img_to_array(x)
            if case > 0:
                x = image_preprocess(x, case)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            img_list.append(x)
        img = np.reshape(np.stack(img_list), [-1,299,299,3])
        # predict with preprogressing in reader
        return img

class Config():
    batch_size = 32
    # path
    root_path = "../planet/"
    imgs_path = root_path + "test-jpg/"
    labels_file = root_path + "sample_submission_v2.csv"
    usecols = range(1,18)
    number = 61191
# reload fine-tune model
model = load_model("./model/inception_v3_1_7.h5")
# set config
config = Config()

labels = ['tags','agriculture', 'artisinal_mine', 'bare_ground', 'blooming', 'blow_down', 'clear',
              'cloudy', 'conventional_mine', 'cultivation', 'habitation', 'haze', 'partly_cloudy',
              'primary', 'road', 'selective_logging', 'slash_burn', 'water']
labels = np.array(labels)[config.usecols].tolist()

imgnames = pd.read_csv(config.labels_file, skiprows=[0], usecols=[0], header=None).values

# load validation set
threshold = [[0.08, 0.02, 0.03, 0.05, 0.02, 0.05, 0.06, 0.01, 0.04, 0.05, 0.06, 0.04, 0.1, 0.07, 0.08, 0.02, 0.06],
            [0.08, 0.01, 0.02, 0.07, 0.02, 0.02, 0.05, 0.01, 0.05, 0.04, 0.06, 0.04, 0.06, 0.06, 0.07, 0.05, 0.05],
            [0.07, 0.02, 0.03, 0.06, 0.01, 0.04, 0.05, 0.01, 0.04, 0.04, 0.06, 0.04, 0.07, 0.07, 0.09, 0.04, 0.07],
            [0.07, 0.01, 0.03, 0.07, 0.01, 0.04, 0.16, 0.01, 0.05, 0.03, 0.07, 0.04, 0.11, 0.05, 0.08, 0.04, 0.05],
            [0.06, 0.01, 0.04, 0.05, 0.01, 0.05, 0.03, 0.01, 0.05, 0.04, 0.07, 0.05, 0.05, 0.08, 0.06, 0.03, 0.05],
            [0.06, 0.01, 0.04, 0.14, 0.01, 0.03, 0.08, 0.01, 0.05, 0.04, 0.07, 0.03, 0.11, 0.07, 0.09, 0.04, 0.05]]
final_out = []
img_reader = read_img(config)
for case in xrange(6):
    pred = []
    img_reader.reset()
    print "preding", case
    iter_num = np.int32(np.ceil(config.number/float(config.batch_size)))
    for i in  xrange(iter_num):
        img = img_reader.read_test_img(case)
        prob = model.predict(img)
        pred = np.append(pred, prob)
        print i , '/' , iter_num

    pred = np.reshape(pred,[-1, len(config.usecols)])

    thres = threshold[case]

    pred_out = np.zeros_like(pred)
    # predict output
    for i in range(len(config.usecols)):
        pred_out[:, i] = (pred[:, i] > thres[i]).astype(np.int)
    if case == 0:
        final_out = pred_out
    else:
        final_out = final_out + pred_out

# vote
final_out = (final_out > 3).astype(np.int)

synset = [l.strip() for l in open("../planet/synset.txt").readlines()]    
print "writing data"
# writting data
with open('./result.csv', 'wb') as f:
    writer = csv.writer(f)
    writer.writerow(['image_name', 'tags'])
    for i in xrange(config.number):
        tag = ""
        for j in xrange(len(config.usecols)):
            if final_out[i,j] == 1:
                tag = tag + synset[j] + ' '
        row = [imgnames[i][0]] + [tag[:-1]]
        writer.writerow(row)
