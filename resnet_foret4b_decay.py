# USAGE
# python resnet_foret4b_decay.py
# set the matplotlib backend so figures can be saved in the background
import matplotlib as plt

plt.use("Agg")

# import the necessary packages
import os
from config import config_resnet_foret4b_decay as cfg
from sklearn.metrics import classification_report, confusion_matrix, recall_score
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from horoma.nn.conv import ResNet4b
from horoma.callbacks import TrainingMonitor, EpochCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD
import tensorflow as tf
from keras.utils import multi_gpu_model
from keras_radam import RAdam
from keras_lookahead import Lookahead
from keras.datasets import cifar10
import numpy as np
import h5py

def poly_decay(epoch):
    # initialize the maximum number of epochs, base learning rate,
    # and power of the polynomial
    maxEpochs = cfg.NUM_EPOCHS
    baseLR = cfg.INIT_LR
    power = 1.2

    # compute the new learning rate based on polynomial decay
    alpha = baseLR * (1 - (epoch / float(maxEpochs))) ** power

    # return the new learning rate
    return alpha

# ################  GPU- SELECTION #########################
# allow to select a specific GPU (for single GPU run only)
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
# os.environ["CUDA_VISIBLE_DEVICES"]="1";  # select GPU : 1

# #######################################################
# load the training and testing data, converting the images from
# integers to floats

f = h5py.File(cfg.filename, 'r')

# #################################################################
# ################  HDF% read #####################################
# #################################################################
print("[INFO] loading forestry data from: ", cfg.filename)
db = np.empty((0, 3, 32, 32))
labels = []
labels_ = []
data = []
data_img = []
data_dsm = []
data_img_ = []
data_dsm_ = []
for regions in f.keys():
    print('[INFO] reading region name :', regions)
    data_group = f[regions]
    if regions == 'autre':
        print('[DEBUG] this regions data is EXCLUDED: ', regions)
        pass
    else:
        for key in data_group.keys():
            # print('INFO] reading dataset name :', key)
            if key == '1_images_DSM':
                print('[INFO] reading dataset name :', key)
                data_dsm_ = np.array(data_group[key][:])
                # print('[DEBUG] data shape', data_dsm_.shape)
            if key == '1_images_IRG':
                print('[INFO] reading dataset name :', key)
                data_img_ = np.array(data_group[key][:])
                # print('[DEBUG] data shape', data_img_.shape)
            if key == '2_essences':
                # print('INFO] reading dataset name :', key)
                labels_ = np.array(data_group[key][:])
                # print('label shape', labels_)
    data_dsm.append(data_dsm_)
    data_img.append(data_img_)
    labels.append(labels_)
    # data.append(data_)
    # labels.append(labels_)
    # print('data lenght', len(data))

data_dsm_merged = np.concatenate(data_dsm, axis=0)
data_dsm_merged = np.expand_dims(data_dsm_merged, axis=1)
data_img_merged = np.concatenate(data_img, axis=0)
# print('[DEBUG] dsm_merged lenght', data_dsm_merged.shape)
# print('[DEBUG] img_merged lenght', data_img_merged.shape)


data = np.hstack((data_img_merged, data_dsm_merged))
labels_merged = np.concatenate(labels, axis=0)
# print('[DEBUG] merged IMG-DSM lenght', data.shape)
# print('[DEBUG] merged labels_merged lenght', len(labels_merged))

# print('merged labels lenght', len(labels_merged))
# data_merged = np.concatenate((data_img_merged,data_dsm_merged),axis=1)
print('[INFO] number of samples :', len(labels_merged))
f.close()
# ################# NO STRATIFICATION ###################################
# print(labels)htop
# print('labels',len(labels))
# (trainX, testX, trainY, testY) = train_test_split(db, labels, test_size=0.5, random_state=42)
# (trainX, testX, trainY, testY) = train_test_split(data_merged, labels_merged, test_size=0.5, random_state=42)

# #############   STRATIFY SAMPLING  TRAIN & TEST ################
(pre_trainX, testX, pre_trainY, testY) = train_test_split(data, labels_merged, test_size=0.15,
                                                          stratify=labels_merged, shuffle=True, random_state=42)

# print('[DEBUG] pre_trainX', len(pre_trainX))
# print('[DEBUG] pre_trainY', len(pre_trainY))
# print('[DEBUG] testX', len(testX))
# print('[DEBUG] testY', len(testY))
# print(pre_trainY)
# #############   STRATIFY SAMPLING  VALIDATE from TRAIN ################
split = train_test_split(pre_trainX, pre_trainY, test_size=0.15, stratify=pre_trainY, random_state=42)
(trainX, valX, trainY, valY) = split

# print('[DEBUG] trainX', len(trainX))
# print('[DEBUG] trainY', len(trainY))
# print('[DEBUG] valX', len(valX))
# print('[DEBUG] valY', len(valY))


# print("[INFO] loading CIFAR-10 data...")
# ((trainX, trainY), (testX, testY)) = cifar10.load_data()
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
aug = ImageDataGenerator(width_shift_range=0.1, vertical_flip=True, rotation_range=180,
                         height_shift_range=0.1, horizontal_flip=True)
opt = Lookahead(RAdam(min_lr=cfg.INIT_LR_RADAM))
# construct the set of callbacks
# radam I removed LearningRateScheduler(poly_decay) from the callbacks
# figPath = cfg.output_PATH + "{}.png".format(os.getpid())
# jsonPath = cfg.output_PATH + "{}.png".format(os.getpid())
# callbacks = [TrainingMonitor(figPath, jsonPath=jsonPath), LearningRateScheduler(poly_decay)]
# construct the set of callbacks
callbacks = [EpochCheckpoint(cfg.EPOCH_PATH, every=cfg.EPOCH_EVERY, startAt=cfg.EPOCH_START),
             TrainingMonitor(cfg.FIG_PATH + "{}.png".format(os.getpid()), jsonPath=cfg.JSON_PATH + "{}.png".format(os.getpid()), startAt=cfg.EPOCH_START)]

# Single or  MULTI GPU model
# See model.build method in ResNet4b.py file
# ResNet4b.build(width, height, # of bands, # of classes, model depth, filters to learn, regularization value)
# the trainX, valX and testX MUST match ResNet4b(width, height, # of bands) values for model to compile
# in SINGLE GPU, the model is loaded into GPU
# in MULTI GPU, the model is loaded into CPU first and then distributed to GPUs.... much faster
if cfg.NBR_GPU <= 1:
    print('[INFO] training on 1 GPU....')
    model = ResNet4b.build(32, 32, 4, 19, (9, 9, 9), (64, 64, 128, 256), reg=0.0005)
    print("[INFO] summary for base model...")
    model.summary()
else:
    print('[INFO] training on {} GPUs...'.format(cfg.NBR_GPU))
    with tf.device("/cpu:0"):
        # init model, load into CPU, distribute to GPUs
        model = ResNet4b.build(32, 32, 4, 19, (9, 9, 9), (64, 64, 128, 256), reg=0.0005)
    # make model parallel
    model = multi_gpu_model(model, gpus=cfg.NBR_GPU)

# initialize the optimizer and model
print("[INFO] compiling model...")
# selection optimizer
# opt = SGD(lr=cfg.INIT_LRINIT_LR, momentum=0.9)
opt = Lookahead(RAdam(min_lr=cfg.INIT_LR_RADAM))
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=['accuracy'])

# train the network
print("[INFO] training network...")
model.fit_generator(
    aug.flow(trainX, trainY, batch_size=cfg.BATCH_SIZE*cfg.NBR_GPU),
    validation_data=(valX, valY),
    steps_per_epoch=len(trainX) // cfg.BATCH_SIZE*cfg.NBR_GPU, epochs=cfg.NUM_EPOCHS,
    callbacks=callbacks, shuffle=True, verbose=2)

# save the network to disk
print("[INFO] serializing network...")
model.save_weights(cfg.output_PATH + "resnet_foret4b_decay" + cfg.TIMESTR + ".hdf5")


# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=cfg.BATCH_SIZE)
print('[DEBUG] predictions', predictions.argmax(axis=1))
print('[DEBUG] testY', testY.argmax(axis=1))
print('[DEBUG] lb.classes_', lb.classes_)

print('[INFO] Classification report')
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=lb.classes_))

print('[INFO] Confusion matrix')
print(confusion_matrix(np.argmax(testY, axis=1), predictions.argmax(axis=1)),lb.classes_)
# print('[INFO] Confusion matrix_cm_analysis')
# plot_confusion_matrix.cm_analysis(testY.argmax(axis=1), predictions.argmax(axis=1), '_confusion', lb.classes_, ymap=None, figsize=(10,10))
#plot_confusion_matrix.cm_analysis(testY.argmax(axis=1), predictions.argmax(axis=1),
#    'home/jp/PycharmProjects/PyImage/PB_Code/chapter12-resnet/output/confusion.png', np.unique(testY), ymap=None, figsize=(10,10))
#labels=np.unique(y_true)