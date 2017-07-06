import numpy as np
from read_data_extend import Reader, random_batch_generator, batch_generator
class Config():
    batch_size = 16
    
    # path
    root_path = "../planet/"
    imgs_path = root_path + "train-jpg/"
    labels_file = root_path + "train_validation_v2_bin.csv"

    usecols = range(1,18)

#train again
from keras.models import load_model
model = load_model("./model/inception_v3_1_8.h5")

# only train full connection layer
for layer in model.layers:
    layer.trainable = False
for layer in model.layers[-2:]:
    layer.trainalbe = True

train_config = Config()
valid_config = Config()
valid_config.labels_file = valid_config.root_path +  "validation_train_v2_bin.csv"

for layer in model.layers[:25]:
    layer.trainable = False
for layer in model.layers[25:]:
    layer.trainable = True

from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')

for case in xrange(6):
    train_config.case = case
    valid_config.case = case
    model.fit_generator(generator=random_batch_generator(train_config), steps_per_epoch=500, epochs=10, 
                    validation_data=batch_generator(valid_config), 
                    validation_steps=np.int32(np.ceil(4048/float(valid_config.batch_size))))

model.save("./model/inception_v3_2_1.h5")