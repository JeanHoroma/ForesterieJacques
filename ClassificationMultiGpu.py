import os

#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
#os.environ["CUDA_VISIBLE_DEVICES"] = "0";  # select GPU : 1
import numpy as np
import time
import resource
from keras.utils import multi_gpu_model
from PIL import Image
from keras.optimizers import SGD
from horoma.nn.conv import ResNet
import cv2


def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    vect = []
    for y in range(0, image.shape[1], stepSize):
        for x in range(0, image.shape[2], stepSize):
            # yield the current window
            yield x, y, image[:, y:y + windowSize[1], x:x + windowSize[0]]



# std dataset path to foresterie project
PATH_OUTPUT = '/home/jp/PycharmProjects/foresterie/output/classification/'
PATH = '/home/jp/PycharmProjects/foresterie/dataset/'
modele_sauve = '/home/jp/PycharmProjects/foresterie/output/resnet_foret3b_decay.hdf5'
(winW, winH) = (32, 32)
INIT_LR = 1e-1

# image examples: file_types = (".jpg", ".jpeg", ".png", ".DAT", ".tif", ".tiff")
# most known file formats are supported
file_types = (".DAT", ".tif", ".tiff")
imagePath = []
list_Filenames = []
# contains= EXCLUSION: cherche la section contenu par "" et si elle existe dans le nom, le fichier ne sera pas lu
contains = None
# loop over the directory structure
for (rootDir, dirNames, filenames) in os.walk(PATH):
    # loop over the filenames in the current directory
    for filename in filenames:
        # if the contains string is not none and the filename does not contain
        # the supplied string, then ignore the file
        if contains is not None and filename.find(contains) == -1:
            continue
            # determine the file extension of the current file

        ext = filename[filename.rfind("."):].lower()

        # check to see if the file is an image and should be processed
        if file_types is None or ext.endswith(file_types):
            # construct the path to the image and yield it
            list_Filenames.append(filename)
            imagePath.append(os.path.join(rootDir, filename))


# Optimizer, Model compile and Load Weights.....
opt = SGD(lr=INIT_LR, momentum=0.9)
model = ResNet.build(32, 32, 3, 19, (9, 9, 9), (64, 64, 128, 256), reg=0.0005)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
model.load_weights(modele_sauve)
# TROP LENT: hypothese a verifier: modele entraine en simple GPU....
# model = multi_gpu_model(model,gpus=2)
#d b = np.empty((0, 3, 32, 32))
print('[INFO] number of files found : ', len(imagePath))
#print('list_filenames',list_Filenames)
for (i, image) in enumerate(imagePath):
    #print(imagePath)
    img = np.asarray(cv2.imread(image), float)
    # print(image)
    shape = img.shape
    time_start = time.perf_counter()

    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]
    r += np.mean(r)
    g += np.mean(g)
    b += np.mean(b)

    img_rec = np.stack((r, g, b), axis=0)

    # print('img_rec shape', img_rec.shape)
    temp = []
    Y = img_rec.shape[1] -winH +1
    X = img_rec.shape[2] -winW +1
    output_shape = (img_rec.shape[0], Y, X)
    print('[INFO] file path, name : ', imagePath[i])
    print('[INFO] input image shape (ch, y, x)', shape)
    print('[INFO] output image shape (y, x, ch) ', output_shape)
    # print('Y,X',Y,X)
    #vect = []
    vect = []
    count = 1

    # accumulate prediction

    # prediction batch size, limited by amount of GPU RAM (higher == faster)
    buffer_size = 10000
    # limited by amount of CPU RAM available for LIST to NP.ARRAY conversion  (higher == faster)
    # It does NOT have to be a fixed multiple of buffer_size:   ex: vector_size = 25000
    # For 128 GB RAM : vector_size = buffer_size * 300  (approx) @ np.float32
    # For 32 GB RAM : vector_size = buffer_size * 75  (approx)
    vector_size = buffer_size * 300
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
            #print('temp, count, x*y',len(temp), count, x*y)
            #vect = np.concatenate(np.asarray(temp, float), axis=0)
            predictions = model.predict(np.concatenate(np.asarray(temp, np.float32), axis=0), batch_size=buffer_size,verbose=0)
            #print('predction......................', predictions.shape)
            predictions = predictions.argmax(axis=1)+1
            #print('total number of blocs and total', bloc_count_max)
            #print('predction..argmax....................', predictions.shape)

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

    final = np.asarray((np.reshape(pred_acc, (Y, X))), dtype=np.uint8)
    print('final.shape',final.shape)
    #
    # write file to output directory
    # /output/filename.tif
    file_name = PATH_OUTPUT + 'prediction_' + list_Filenames[i]

    image_temp = Image.fromarray(final)
    image_temp.save(file_name)
    print('processing', list_Filenames[i])
    count = 1
    #pred_acc.clear()
    time_elapsed = (time.perf_counter() - time_start)
    memMb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0/1024.0
    print("TIME USED:%5.1f secs      MEMORY USAGE:%5.1f MByte" % (time_elapsed, memMb))
    # print('pred',predictions)
        #  TEST SLIDING WINDOW WITH OVERLAY.....
        # clone = image.copy()
        # cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
        # cv2.imshow("Window", clone)
        # cv2.waitKey(1)
        # time.sleep(0.025)
    # np.concatenate(temp, axis=0)

