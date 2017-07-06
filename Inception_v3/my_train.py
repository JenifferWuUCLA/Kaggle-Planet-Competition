import sys
sys.path.insert(0, '../planet')
from inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K

from read_data import Reader
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

def random_batch_generator(config):
    reader = Reader(config)
    while True:
        batch_features, batch_labels = reader.random_batch()
        yield batch_features, batch_labels
def batch_generator(config):
    reader = Reader(config)
    while True:
        batch_features, batch_labels = reader.batch()
        yield batch_features, batch_labels
        

# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- 17 classes
predictions = Dense(17, activation='softmax')(x)

# this is the model we will train
print "init model"
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
print "compile model"
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

print "fit model's new layer"
train_config = Config()
valid_config = Config()
valid_config.labels_file = valid_config.root_path +  "validation_train_v2_bin.csv"
# train the model on the new data for a few epochs
model.fit_generator(generator=random_batch_generator(train_config), steps_per_epoch=50, epochs=15)

# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
for i, layer in enumerate(base_model.layers):
    print(i, layer.name)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 172 layers and unfreeze the rest:
for layer in model.layers[:172]:
    layer.trainable = False
for layer in model.layers[172:]:
    layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
model.fit_generator(generator=random_batch_generator(train_config), steps_per_epoch=50, epochs=35, 
                    validation_data=batch_generator(valid_config), validation_steps=50)

model.save("./model/inception_1.h5")