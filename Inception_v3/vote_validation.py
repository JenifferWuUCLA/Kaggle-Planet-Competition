import numpy as np
from fbeta_score import f2_score, optimise_f2_thresholds
from sklearn.metrics import accuracy_score
from read_data_extend import Reader
from keras.models import load_model

class Config():
    batch_size = 16
    
    # path
    root_path = "../planet/"
    imgs_path = root_path + "train-jpg/"
    labels_file = root_path + "train_validation_v2_bin.csv"
    
    # iterations config
    max_iteration = 500
    summary_iters = 50
    valid_iters = 250
    usecols = range(1,18)

# reload fine-tune model
model = load_model("./model/inception_v3_1_8.h5")
# set config
valid_config = Config()
valid_config.labels_file = valid_config.root_path +  "validation_train_v2_bin.csv"
valid_config.batch_size = 32
valid_reader = Reader(valid_config)

labels = ['tags','agriculture', 'artisinal_mine', 'bare_ground', 'blooming', 'blow_down', 'clear',
              'cloudy', 'conventional_mine', 'cultivation', 'habitation', 'haze', 'partly_cloudy',
              'primary', 'road', 'selective_logging', 'slash_burn', 'water']
labels = np.array(labels)[valid_config.usecols].tolist()

# load validation set

valid_true_out = []
final_out = []
print "preding..."
for case in xrange(6):
    print "preding", case
    valid_pred = list()
    for x in  xrange(np.int32(np.ceil(4048/float(valid_config.batch_size)))):
        valid_img, valid_label = valid_reader.batch(case)
        # predict with preprogressing in reader
        valid_prob = model.predict(valid_img)
        valid_pred = np.append(valid_pred, valid_prob)
        if case == 0:
            valid_true_out = np.append(valid_true_out, valid_label)

    valid_pred = np.reshape(valid_pred,[-1, len(valid_config.usecols)])
    if case == 0:
        valid_true_out = np.reshape(valid_true_out, [-1, len(valid_config.usecols)])

    # find best threshold
    thres = optimise_f2_thresholds(valid_true_out, valid_pred)
    print thres
    valid_pred_out = np.zeros_like(valid_pred)
    # predict output
    for i in range(len(valid_config.usecols)):
        valid_pred_out[:, i] = (valid_pred[:, i] > thres[i]).astype(np.int)
    if case == 0:
        final_out = valid_pred_out
    else:
        final_out = final_out + valid_pred_out

#vote
final_out = (final_out > 3).astype(np.int)

# compute f beta score
valid_f2_score = f2_score(valid_true_out, final_out)
print "valid f2 score:", valid_f2_score

# compute accuracy
for i in xrange(len(valid_config.usecols)):
    acy_score = accuracy_score(valid_true_out[:, i], final_out[:, i])
    print "acy_score:\t", labels[i], "\t", acy_score
