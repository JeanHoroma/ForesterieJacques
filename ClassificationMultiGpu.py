import os

#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
#os.environ["CUDA_VISIBLE_DEVICES"] = "0";  # select GPU : 1
import numpy as np
import time
from datetime import datetime
import resource
from keras.utils import multi_gpu_model
import tensorflow as tf
from PIL import Image, ImageOps
from keras.optimizers import SGD
from horoma.nn.conv import ResNet
import cv2
import georaster as gr
import gdal


def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    vect = []
    for y in range(0, image.shape[1], stepSize):
        for x in range(0, image.shape[2], stepSize):
            # yield the current window
            yield x, y, image[:, y:y + windowSize[1], x:x + windowSize[0]]


time_stamp = datetime.now().strftime("%d-%m-%Y_%I-%M-%S_%p")
# std dataset path to foresterie project
PATH_OUTPUT = '/home/jp/PycharmProjects/foresterie/output/classification_4b/'
PATH = '/home/jp/PycharmProjects/foresterie/dataset/'
# modele_sauve = '/home/jp/PycharmProjects/foresterie/output/resnet_foret3b_decay2020-04-10.hdf5'
modele_sauve = '/home/jp/PycharmProjects/foresterie/output/classification_4bresnet_foret4b_decay20200412-15h37m.hdf5'
(winW, winH) = (32, 32)
INIT_LR = 1e-1
NBR_GPU = 2

# image examples: file_types = (".jpg", ".jpeg", ".png", ".DAT", ".tif", ".tiff")
# most known file formats are supported
file_types = (".DAT", ".tif", ".tiff")
imagePath = []
list_Filenames = []
imagePathDSM = []
list_FilenamesDSM = []

# contains= cherche str dans nom de fichier
contains = 'dsm'
# loop over the directory structure, create list of all files with the file type list
for (rootDir, dirNames, filenames) in os.walk(PATH):
    # loop over the filenames in the current directory
    for filename in filenames:
        # if the contains string is not none and the filename does contain
        # the supplied string, then select file
        if contains is None:
            continue
        elif filename.find(contains) > -1:
            list_FilenamesDSM.append(filename)
            imagePathDSM.append(os.path.join(rootDir, filename))

        # determine the file extension of the current file

        ext = filename[filename.rfind("."):].lower()

        # check to see if the file is an image and should be processed
        if file_types is None or ext.endswith(file_types):
            # construct the path to the image and yield it
            list_Filenames.append(filename)
            imagePath.append(os.path.join(rootDir, filename))

imagePath.sort()
list_Filenames.sort()
imagePathDSM.sort()
list_FilenamesDSM.sort()

# Optimizer, Model compile and Load Weights.....
opt = SGD(lr=INIT_LR, momentum=0.9)
with tf.device("/cpu:0"): # doit loader modele dans CPU pour s'assurer que param sont distribue a partir du CPU et non GPU
    # init model
    model = ResNet.build(32, 32, 3, 19, (9, 9, 9), (64, 64, 128, 256), reg=0.0005)
model_para = multi_gpu_model(model, gpus=2)

model_para.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
model_para.load_weights(modele_sauve)
# TROP LENT: hypothese a verifier: modele entraine en simple GPU....

#d b = np.empty((0, 3, 32, 32))
print('[INFO] number of files found : ', len(imagePath))
#print('list_filenames',list_Filenames)
for (i, image) in enumerate(imagePath):
    #print(imagePath)
    img = np.asarray(cv2.imread(image), dtype=np.float32)
    dsm_file = np.asarray(cv2.imread(imagePathDSM[i]), dtype=np.float32)
    # print(image)
    shape = img.shape
    time_start = time.perf_counter()

    r = img[:, :, 0].copy()
    g = img[:, :, 1].copy()
    b = img[:, :, 2].copy()
    d = dsm_file.copy()
    #r += np.mean(r)
    #g += np.mean(g)
    #b += np.mean(b)

    img_rec = np.stack((r, g, b, d), axis=0)

    # print('img_rec shape', img_rec.shape)
    temp = []
    Y = img_rec.shape[1] -winH +1
    X = img_rec.shape[2] -winW +1
    output_shape = (img_rec.shape[0], Y, X)
    print('[INFO] file path, name : ', imagePath[i])
    print('[INFO] input image shape (ch, y, x)', shape)
    print('[INFO] processed image shape (y, x, ch) ', output_shape)
    # print('Y,X',Y,X)
    #vect = []
    vect = []
    count = 1

    # accumulate prediction

    # prediction batch size, limited by amount of GPU RAM (higher == faster)
    buffer_size = 8192 * NBR_GPU
    # ########### VALID FOR RGB ONLY #########################
    # limited by amount of CPU RAM available for LIST to NP.ARRAY conversion  (higher == faster)
    # It does NOT have to be a fixed multiple of buffer_size:   ex: vector_size = 25000
    # For 128 GB RAM : vector_size = buffer_size * 300  (approx) @ np.float32
    # For 32 GB RAM : vector_size = buffer_size * 70  (approx)
    # If problem, (SIGKILL = 9) , verify memory allocation with htop in shell
    vector_size = buffer_size * 300 / NBR_GPU
    pred_acc = []
    bloc_count_max = round(X * Y /(vector_size))
    bloc_count_start =1
    for (x, y, window) in sliding_window(img_rec, stepSize=1, windowSize=(winW, winH)):
        # if the window does not meet our desired window size, ignore it
        if window.shape[1] != winH or window.shape[2] != winW:
            continue

        # print('y,x,count', y, x,count)
        # print('windows_shape',window.shape)
        # time.sleep(0.1)

        if count == 0:
            temp.append([window])
            #print('temp',len(temp), count)
            count += 1
        elif count % vector_size == 0 or count == X*Y: # if buffer is FULL = vector_size OR end of file size = Y*X
            temp.append([window])
            # print('temp, count, x*y',len(temp), count, x*y)
            vect = np.concatenate(np.asarray(temp, dtype=np.float32), axis=0)
            mean = np.mean(vect, axis=0)
            vect -= mean
            predictions = model_para.predict(vect, batch_size=buffer_size,verbose=0)
            # print('predction......................', predictions.shape)
            predictions = predictions.argmax(axis=1)+1
            # print(predictions)
            # update counter for STDOUT
            # print("Image Vector Progress {:02d} of {:02d}".format(bloc_count_start, bloc_count_max), end="\r")
            # bloc_count_start += 1
            # print('total number of blocs and total', bloc_count_max)
            # print('predction..argmax....................', predictions.shape)

            if count <= vector_size: # first iteration, cannot do np.concatenate()
                pred_acc = predictions
            else:
                #
                pred_acc = np.concatenate((pred_acc, predictions), axis=0)

            # Clear buffer to ZERO, ready for next accumulation
            temp.clear()
            count += 1

            # print('clear... temps', len(temp))
        else:
            temp.append([window])
            #print('temp_else',len(temp),count)
            count += 1

    final = np.asarray((np.reshape(pred_acc, (Y, X))), np.float32)

    #
    # write file to output directory
    # /output/filename.tif
    file_name = PATH_OUTPUT + 'pred_' + list_Filenames[i]
    border = (16,16,15,15)
    image_temp = Image.fromarray(final)
    image_temp1 = np.asarray(ImageOps.expand(image_temp,border)) # add border to resize to original value,convert to array, again
    # print('final.shape', image_temp1.size)
    # image_temp1.save(file_name)
    # Insert GEOTIFF information back into image before saving
    info_geo = gr.MultiBandRaster(imagePath[i], load_data=False)
    im_gen = gr.SingleBandRaster.from_array(image_temp1,info_geo.trans,info_geo.proj.srs,gdal.GDT_Int16)
    im_gen.save_geotiff(PATH_OUTPUT + 'geo_pred_' + list_Filenames[i])
    print('processing', list_Filenames[i])
    count = 1
    #pred_acc.clear()
    time_elapsed = (time.perf_counter() - time_start)
    HMS_time = time.strftime("%H:%M:%S", time.gmtime(time_elapsed))
    memMb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0/1024.0
    print("[INFO] TIME USED: ", HMS_time)
    print("[INFO] MEMORY USAGE: {:.1f} MByte".format(memMb))
    # print('pred',predictions)
        #  TEST SLIDING WINDOW WITH OVERLAY.....
        # clone = image.copy()
        # cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
        # cv2.imshow("Window", clone)
        # cv2.waitKey(1)
        # time.sleep(0.025)
    # np.concatenate(temp, axis=0)

