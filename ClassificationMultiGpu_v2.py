import os
# two lines below used to select specific GPU for tests purposes
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
# os.environ["CUDA_VISIBLE_DEVICES"] = "0";  # select GPU : 1
from horoma.nn.conv import ResNet4b
from horoma.io import sliding_window, ListFiles
from config import config_ClassificationMultiGpu_v2 as cfg
import numpy as np
import time
import resource
from keras.utils import multi_gpu_model
import tensorflow as tf
from PIL import Image, ImageOps
from keras.optimizers import SGD
import cv2
import georaster as gr
import gdal

# image examples: file_types = (".jpg", ".jpeg", ".png", ".DAT", ".tif", ".tiff")
# most known file formats are supported
file_types = (".DAT", ".tif", ".tiff")

# loop over the directory structure, create list of all files with the file type list
# select .tif files
# contains = look for specific str in filename. Example = 'dsm'
info_dsm = ListFiles(PATH=cfg.PATH, contains='dsm', avoid=None, file_types=file_types)
info_file = ListFiles(PATH=cfg.PATH, contains=None, avoid='dsm', file_types=file_types)

imagePath = info_file[0]
list_Filenames = info_file[1]
imagePathDSM = info_dsm[0]
list_FilenamesDSM = info_dsm[1]

# print('[DEBUG] imagepath', info_dsm)
# print('[DEBUG] imagepathdsm', imagePathDSM)
# print('[DEBUG] imagepath LISTE', list_Filenames)
# print('[DEBUG] imagepathdsm LISTE--DSM', list_FilenamesDSM)
imagePath.sort()
list_Filenames.sort()
imagePathDSM.sort()
list_FilenamesDSM.sort()

print('[INFO] number of files found : ', len(imagePath))

# Optimizer, Model compile and Load Weights.....
# tf.device force to load model into CPU for distribution to GPU
opt = SGD(lr=cfg.INIT_LR, momentum=0.9)
with tf.device("/cpu:0"):
    # init model
    model = ResNet4b.build(32, 32, 4, 19, (9, 9, 9), (64, 64, 128, 256), reg=0.0005)
model_para = multi_gpu_model(model, gpus=2)
model_para.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
model_para.load_weights(cfg.modele_sauve)

for (i, image) in enumerate(imagePath):
    # time measurement of loading and processing each file
    time_start = time.perf_counter()

    # load the IRG band from dataset (band 0,1,2)
    img = np.asarray(cv2.imread(image), dtype=np.float32)
    # load DSM band from dataset band(3)
    dsm_file = np.asarray(cv2.imread(imagePathDSM[i], -1), dtype=np.float32)
    # add any band necessary to train model

    # merge all bands together before stacking into np.array (along axis = 0)
    r = img[:, :, 0].copy()
    g = img[:, :, 1].copy()
    b = img[:, :, 2].copy()
    d = dsm_file.copy()
    # d = dsm_file[:, :, 0].copy()

    img_rec = np.stack((r, g, b, d), axis=0)
    # print('[DEBUG] img-rec.shape..................', img_rec.shape)
    # print('i[DEBUG] mg_rec shape', img_rec.shape)
    temp = []
    Y = img_rec.shape[1] - cfg.winH + 1
    X = img_rec.shape[2] - cfg.winW + 1
    output_shape = (img_rec.shape[0], Y, X)
    # print('[DEBUG] file path IRG, name : ', imagePath[i])
    # print('[DEBUG] file path DSM, name : ', imagePathDSM[i])
    # print('[DEBUG] input image shape (ch, y, x)', img_rec.shape)
    # print('[DEBUG] processed image shape (y, x, ch) ', output_shape)
    # print('Y,X',Y,X)

    vect = []
    # prediction batch size, limited by amount of GPU RAM (higher == faster)
    buffer_size = 8192 * cfg.NBR_GPU
    # ########### VALID FOR RGB ONLY #########################
    # limited by amount of CPU RAM available for LIST to NP.ARRAY conversion  (higher == faster)
    # It does NOT have to be a fixed multiple of buffer_size:   ex: vector_size = 25000
    # For 128 GB RAM : vector_size = buffer_size * 200  (approx) @ np.float32, 4 bandes
    # For 32 GB RAM : vector_size = buffer_size * 70  (approx)
    # If problem, (SIGKILL = 9) , verify memory allocation with htop in shell
    vector_size = buffer_size * 250 / cfg.NBR_GPU
    pred_acc = []
    bloc_count_max = round(X * Y / vector_size)
    bloc_count_start = 1
    count = 0
    for (x, y, window) in sliding_window(img_rec, stepSize=1, windowSize=(cfg.winW, cfg.winH)):
        # if the window does not meet our desired window size, ignore it

        if window.shape[1] != cfg.winH or window.shape[2] != cfg.winW:
            continue

        # print('y,x,count', y, x,count,vector_size, X * Y)
        # print('windows_shape',window.shape)
        # time.sleep(0.1)

        if count == 0:
            temp.append([window])
            # print('temp',len(temp), count)
            count += 1
        elif count % vector_size == 0 or count == X * Y - 1:  # if buffer is FULL = vector_size OR end of file size = Y*X
            temp.append([window])
            # print("Image Vector Progress {:02d} of {:02d}".format(bloc_count_start, bloc_count_max), end='\r', flush=True)
            # bloc_count_start += 1
            # print('temp, count, x*y',len(temp), count, x*y)
            vect = np.concatenate(np.asarray(temp, dtype=np.float32), axis=0)
            mean = np.mean(vect, axis=0)
            vect -= mean
            predictions = model_para.predict(vect, batch_size=buffer_size, verbose=0)
            predictions = predictions.argmax(axis=1) + 1

            if count <= vector_size:
                # first iteration, cannot do np.concatenate()
                pred_acc = predictions

            else:
                pred_acc = np.concatenate((pred_acc, predictions), axis=0)

            # Clear buffer to ZERO, ready for next accumulation
            temp.clear()
            count += 1

            # print('clear... temps', len(temp))
        else:
            temp.append([window])
            # print('temp_else',len(temp),count)
            count += 1
    # print('pred_acc',len(pred_acc))
    final = np.asarray((np.reshape(pred_acc, (Y, X))), np.float32)
    # write file to output directory
    # /output/filename.tif
    # file_name = cfg.PATH_OUTPUT + 'pred_' + list_Filenames[i]
    border = (16, 16, 15, 15)
    image_temp = Image.fromarray(final)
    # add border to resize to original value,convert to array, again
    image_temp1 = np.asarray(ImageOps.expand(image_temp, border))
    # print('[dfinal.shape', image_temp1.size)
    # image_temp1.save(file_name)
    # Insert GEOTIFF information back into image before saving
    info_geo = gr.MultiBandRaster(imagePath[i], load_data=False)
    im_gen = gr.SingleBandRaster.from_array(image_temp1, info_geo.trans, info_geo.proj.srs, gdal.GDT_Int16)
    im_gen.save_geotiff(cfg.PATH_OUTPUT + 'geo_pred_4b_' + list_Filenames[i])
    print('[INFO] File Processed...', list_Filenames[i])
    # count = 1
    # pred_acc.clear()
    # time and memory measurement for each file processed
    time_elapsed = (time.perf_counter() - time_start)
    HMS_time = time.strftime("%H:%M:%S", time.gmtime(time_elapsed))
    memMb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 / 1024.0
    print("[INFO] PROCESSING TIME : ", HMS_time)
    print("[INFO] PROCESSING MEMORY : {:.1f} MByte".format(memMb))
