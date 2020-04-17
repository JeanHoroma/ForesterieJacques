# USAGE
# python foret_google_net.py --output output --model output/minigooglenet_foret.hdf5

# set the matplotlib backend so figures can be saved in the background
import matplotlib

#matplotlib.use("Agg")


# import the necessary packages
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"]="0";  # select GPU : 1
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from horoma.nn import MiniGoogLeNet
from horoma.callbacks import TrainingMonitor
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD
from keras.datasets import cifar10
import numpy as np
import argparse
import h5py
import keras
keras.backend.set_image_data_format('channels_first')
# definine the total number of epochs to train for along with the
# initial learning rate
NUM_EPOCHS = 70
INIT_LR = 5e-2


def poly_decay(epoch):
    # initialize the maximum number of epochs, base learning rate,
    # and power of the polynomial
    maxEpochs = NUM_EPOCHS
    baseLR = INIT_LR
    power = 1.0

    # compute the new learning rate based on polynomial decay
    alpha = baseLR * (1 - (epoch / float(maxEpochs))) ** power

    # return the new learning rate
    return alpha


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, help="path to output model")
ap.add_argument("-o", "--output", required=True, help="path to output directory (logs, plots, etc.)")
args = vars(ap.parse_args())

# load the training and testing data, converting the images from
# integers to floats
#print("[INFO] loading HDF5_foret data...")
filename = 'dataset/20200323_foresterie.hdf5'
f = h5py.File(filename, 'r')
#
db = np.empty((0, 3, 32, 32))
labels = []
for key in f.keys():
    print(key)
    group = f[key]
    if key == 'LANAUDIERE':
        pass
    else:
        for key in group.keys():
            if key == '1_images_IRG':
                temp = np.array(group[key][:])
                db = np.append(db, temp, axis=0)
                print(db.shape)
                print(key)
            if key == '2_essences':
                temp = (group[key][()])
                labels = np.append(labels, temp, axis=0)
                print(labels)
                print(key)
f.close()
#print(labels)
#print('labels',len(labels))
(trainX, testX, trainY, testY) = train_test_split(db, labels, test_size=0.3, random_state=42)

#(trainX, testX, trainY, testY) = train_test_split(db, labels, test_size=0.05, random_state=42)
#((trainX, trainY), (testX, testY)) = cifar10.load_data()
#print('ttrainx',ttrainX.shape)
#print('ttrainy',ttrainY)
#print('ttestx',ttrainX)
#print('ttesty',ttrainX)
trainX = trainX.astype("float")/255
testX = testX.astype("float")/255
print('trainX',trainX.shape)
#R=[]
#for i in range(len(labels)-1):
#     R.append(np.mean((trainX[i,:0,::])))
##    #G = trainX[i,:1,::]
##    #B = trainX[i,:2,::]
#
#print('R',R)
#print('G',G.shape)
#print('mean',np.mean(R))
# apply mean subtraction to the data
#mean = np.mean(trainX, axis=0)
#trainX -= mean
#testX -= mean

print('trainX',testY.shape)
# convert the labels from integers to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=30, zoom_range=[0.9, 1.1], shear_range=5.0,
                         vertical_flip=True, horizontal_flip=True)

# construct the set of callbacks
figPath = os.path.sep.join([args["output"], "{}.png".format(os.getpid())])
jsonPath = os.path.sep.join([args["output"], "{}.json".format(os.getpid())])
callbacks = [TrainingMonitor(figPath, jsonPath=jsonPath),LearningRateScheduler(poly_decay)]

# initialize the optimizer and model
print("[INFO] compiling model...")
opt = SGD(lr=INIT_LR, momentum=0.9)
model = MiniGoogLeNet.build(width=32, height=32, depth=3, classes=19)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# train the network
print("[INFO] training network...")
model.fit_generator(aug.flow(trainX, trainY, batch_size=100),
                    validation_data=(testX, testY), steps_per_epoch=len(trainX) // 100,
                    epochs=NUM_EPOCHS, callbacks=callbacks, verbose=1)

# save the network to disk
print("[INFO] serializing network...")
model.save(args["model"])
