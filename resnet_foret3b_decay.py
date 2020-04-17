# USAGE
# python resnet_foret3b_decay.py
# fichier config repertoire de sortie
# savegarde du modele: /output/ resnet_foret3b_decay_DATE.AUJOURD'HUI.Heure.Minute.dhf5

# set the matplotlib backend so figures can be saved in the background
import clf as clf
import matplotlib as plt

plt.use("Agg")

# import the necessary packages
from config import config_resnet_foret3b_decay as cfg
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from horoma.nn.conv import ResNet
from horoma.callbacks import TrainingMonitor
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD
from keras.utils import multi_gpu_model
import tensorflow as tf
from keras.datasets import cifar10
import numpy as np
import os
import h5py

# define the total number of epochs to train for along with the
# initial learning rate
NUM_EPOCHS = cfg.NUM_EPOCHS
INIT_LR = cfg.INIT_LR
NBR_GPU = cfg.NBR_GPU
BATCH = cfg.BATCH_SIZE

def poly_decay(epoch):
    # initialize the maximum number of epochs, base learning rate,
    # and power of the polynomial
    maxEpochs = NUM_EPOCHS
    baseLR = INIT_LR
    power = 1.1
    # compute the new learning rate based on polynomial decay
    alpha = baseLR * (1 - (epoch / float(maxEpochs))) ** power
    # return the new learning rate
    return alpha

# load data, see config file for location
f = h5py.File(cfg.filename, 'r')

# #################################################################
# ################  HDF% read #####################################
# #################################################################
print("[INFO] loading configuration from: ", cfg.filename)
db = np.empty((0, 3, 32, 32))
labels = []
data = []
data_ = []
labels_ = []
for regions in f.keys():
    print('[INFO] reading region name :', regions)
    data_group = f[regions]
    if regions == 'non' or regions == 'non':
        print('[DEBUG pass region : ', regions)
        pass
    else:
        for key in data_group.keys():
            # print('INFO] reading dataset name :', key)
            if key == '1_images_IRG':
                print('[INFO] reading dataset name :', key)
                data_ = np.array(data_group[key][:])
                # print('[DEBUG] data shape', data_.shape)
            if key == '2_essences':
                print('[INFO] reading dataset name :', key)
                labels_ = np.array(data_group[key][:])
                # print('[DEBUG] label shape', labels_)
    data.append(data_)
    labels.append(labels_)
    # print('[DEBUG] data lenght', len(data))
data_merged = np.concatenate(data, axis=0)
# print('[DEBUG] merged data lenght', len(data_merged))
labels_merged = np.concatenate(labels, axis=0)
# print('[DEBUG] merged labels lenght', len(labels_merged))
f.close()

# #############   STRATIFY SAMPLING  TRAIN & TEST ################
(pre_trainX, testX, pre_trainY, testY) = train_test_split(data_merged, labels_merged, test_size=0.1,
                                                          stratify=labels_merged, shuffle=True, random_state=42)

# print('[DEBUG] pre_trainX', len(pre_trainX))
# print('[DEBUG] pre_trainY', len(pre_trainY))
# print('[DEBUG] testX', len(testX))
# print('[DEBUG] testY', len(testY))
# print(pre_trainY)

# #############   STRATIFY SAMPLING  VALIDATE from TRAIN ################
split = train_test_split(pre_trainX, pre_trainY, test_size=0.1, stratify=pre_trainY, random_state=42)
(trainX, valX, trainY, valY) = split

# print('[DEBUG] trainX', len(trainX))
# print('[DEBUG] trainY', len(trainY))
# print('[DEBUG] valX', len(valX))
# print('[DEBUG] valY', len(valY))

# ####################### DEBUG RESNET ###################################
# FOR DEBUG PURPOSES, load CIFAR-10 (image format is 32x32, same as foresterie dataset
# CIFAR-10 has 10 classes and 60000 labelled images
# print("[INFO] loading CIFAR-10 data...")
# ((trainX, trainY), (testX, testY)) = cifar10.load_data()
# ########################################################################
trainX = trainX.astype(np.float32)
testX = testX.astype(np.float32)
valX = valX.astype(np.float32)
# apply mean subtraction to the data
mean = np.mean(trainX, axis=0)

trainX -= mean
testX -= mean
valX -= mean
# convert the labels from integers to vectors
le = LabelEncoder
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)
valY = lb.fit_transform(valY)
print('[DEBUG] lb_classes_', lb.classes_)
print('[DEBUG] trainY', trainY.shape)
print('[DEBUG] testY', testY.shape)

# save label classes to file essences.txt
f = open(cfg.output_PATH + "essences.txt", "w+")
f.write(str(lb.classes_))
f.close()
# construct the image generator for data augmentation
# adding rotation=20 decrease val_accuracy
aug = ImageDataGenerator(width_shift_range=0.1, vertical_flip=True,
                         height_shift_range=0.1, horizontal_flip=True)

# construct the set of callbacks
figPath = cfg.output_PATH + "{}.png".format(os.getpid())
jsonPath = cfg.output_PATH + "{}.png".format(os.getpid())
callbacks = [TrainingMonitor(figPath, jsonPath=jsonPath), LearningRateScheduler(poly_decay)]

# initialize the optimizer and model (ResNet-56)
print("[INFO] compiling model...")
opt = SGD(lr=INIT_LR, momentum=0.9)

# Single GPU or MULTI GPU model

if NBR_GPU <= 1:
    print('[INFO] training on 1 GPU....')
    model = ResNet.build(32, 32, 3, 19, (9, 9, 9), (64, 64, 128, 256), reg=0.0005)
    print("[INFO] summary for base model...")
    model.summary()
else:
    print('[INFO] training on {} GPUs...'.format(NBR_GPU))
    # LOAD PARAMETERS INTO CPU, then send values to GPU
    # Faster to load model on CPU , then distribute to GPU
    with tf.device("/cpu:0"):
        # init model
        model = ResNet.build(32, 32, 3, 19, (9, 9, 9), (64, 64, 128, 256), reg=0.0005)
    # make model parallel
    model = multi_gpu_model(model, gpus=NBR_GPU)
    print("[INFO] summary for base model...")
    model.summary()

model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# train the network
print("[INFO] training network...")
model.fit_generator(
    aug.flow(trainX, trainY, batch_size=BATCH * NBR_GPU),
    validation_data=(valX, valY),
    steps_per_epoch=len(trainX) // BATCH * NBR_GPU, epochs=NUM_EPOCHS,
    callbacks=callbacks, verbose=2)

# save the network to disk
print("[INFO] serializing network...")
# model.save_weights(args["model"])
model.save_weights(cfg.output_PATH + "resnet_foret3b_decay" + cfg.TIMESTR + ".hdf5")

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=BATCH)
print('[DEBUG] predictions', predictions.argmax(axis=1))
print('[DEBUG] testY', testY.argmax(axis=1))
print('[DEBUG] lb.classes_', lb.classes_)

print('[INFO] Classification report')
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=lb.classes_))

print('[INFO] Confusion matrix')
print(confusion_matrix(np.argmax(testY, axis=1), predictions.argmax(axis=1)))
# print('[INFO] Confusion matrix_cm_analysis')
# plot_confusion_matrix.cm_analysis(testY.argmax(axis=1), predictions.argmax(axis=1), '_confusion', lb.classes_, ymap=None, figsize=(10,10))
# plot_confusion_matrix.cm_analysis(testY, predictions, 'confusion', labels, ymap=None, figsize=(10,10))
