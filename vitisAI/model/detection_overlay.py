
import cv2, os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys, time, warnings
from datetime import datetime #DB
import pandas as pd

def give_color_to_seg_img(seg,n_classes):
    if len(seg.shape)==3:
        seg = seg[:,:,0]
    seg_img = np.zeros( (seg.shape[0],seg.shape[1],3) ).astype('float')
    #colors = sns.color_palette("hls", n_classes) #DB
    colors = cfg.COLORS #DB
    for c in range(n_classes):
        segc = (seg == c)
        seg_img[:,:,0] += (segc*( colors[c][0]))
        seg_img[:,:,1] += (segc*( colors[c][1]))
        seg_img[:,:,2] += (segc*( colors[c][2]))

    return(seg_img)


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.backend               import set_session
from tensorflow.keras                       import backend #, models
from tensorflow.keras.models                import load_model
from tensorflow.keras.utils                 import plot_model #DB

import gc #DB
from config import fcn_config as cfg
from config import fcn8_cnn as cnn

warnings.filterwarnings("ignore")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.85
config.gpu_options.visible_device_list = "0"
set_session(tf.compat.v1.Session(config=config))

HEIGHT = cfg.HEIGHT
WIDTH  = cfg.WIDTH
N_CLASSES = cfg.NUM_CLASSES
EPOCHS = cfg.EPOCHS
UPSCALE=False

dir_data = cfg.DATASET_DIR
dir_train_img = cfg.dir_train_img
dir_train_seg = cfg.dir_train_seg
dir_test_img  = cfg.dir_test_img
dir_test_seg  = cfg.dir_test_seg
dir_calib_img = cfg.dir_calib_img
dir_calib_seg = cfg.dir_calib_seg
dir_valid_img = cfg.dir_valid_img
dir_valid_seg = cfg.dir_valid_seg

######################################################################
# model
######################################################################

#model = UNET((HEIGHT, WIDTH, 3))

model = cnn.FCN8(nClasses     = N_CLASSES,
             input_height = HEIGHT,
             input_width  = WIDTH,
                 upscale = UPSCALE)


######################################################################
# prepare testing and validation data
######################################################################

# load validation images
valid_images = os.listdir(dir_valid_img)
valid_images.sort()
valid_segmentations  = os.listdir(dir_valid_seg)
valid_segmentations.sort()
X_valid = []
Y_valid = []
for im , seg in zip(valid_images,valid_segmentations) :
    X_valid.append( cnn.NormalizeImageArr(os.path.join(dir_valid_img, im)) )
    Y_valid.append( cnn.LoadSegmentationArr( os.path.join(dir_valid_seg, seg), N_CLASSES , WIDTH, HEIGHT)  )
X_valid, Y_valid = np.array(X_valid) , np.array(Y_valid)
print("\n")
print("validation data (X) (Y) shapes:", X_valid.shape,Y_valid.shape)

test_images = os.listdir(dir_test_img)
test_images.sort()
test_segmentations  = os.listdir(dir_test_seg)
test_segmentations.sort()
X_test = []
Y_test = []
for im , seg in zip(test_images,test_segmentations) :
    X_test.append( cnn.NormalizeImageArr(os.path.join(dir_test_img,im)) )
    Y_test.append( cnn.LoadSegmentationArr(os.path.join(dir_test_seg, seg), N_CLASSES , WIDTH, HEIGHT)  )
X_test, Y_test = np.array(X_test) , np.array(Y_test)
print("testing    data (X) (Y) shapes", X_test.shape,Y_test.shape)
print("\n")

#########################################################################################################
# Calculate intersection over union for each segmentation class

if UPSCALE==False :
    model_filename= "../keras_model/fcn8/ep"    + str(EPOCHS) + "_trained_fcn8_"    + str(WIDTH) + "x" + str(HEIGHT) + ".hdf5"
else :
    model_filename= "../keras_model/fcn8ups/ep" + str(EPOCHS) + "_trained_fcn8ups_" + str(WIDTH) + "x" + str(HEIGHT) + ".hdf5"

model = load_model(model_filename) #DB

print("\nnow computing IoU over testing data set:")
y_pred1   = model.predict(X_test)
y_pred1_i = np.argmax(y_pred1, axis=3)
y_test1_i = np.argmax(Y_test, axis=3)
#print(y_test1_i.shape,y_pred1_i.shape)

np.save("ref_y_pred.npy",   y_pred1)
np.save("ref_y_pred_i.npy", y_pred1_i)
np.save("ref_y_test.npy",   Y_test)
np.save("ref_y_test_i.npy", y_test1_i)

cnn.IoU(y_test1_i, y_pred1_i)

print("\nnow computing IoU over validation data set:")
y_pred2 = model.predict(X_valid)
y_pred2_i = np.argmax(y_pred2, axis=3)
y_test2_i = np.argmax(Y_valid, axis=3)
#print(y_test2_i.shape,y_pred2_i.shape)
cnn.IoU(y_test2_i,y_pred2_i)
image_path='/workspace/Vitis-AI-Tutorials-2.5/Tutorials/Keras_FCN8_UNET_segmentation/files/code/training_32.png'
img2=cv2.imread(image_path,1)
cv2.imshow("iamge",img2)
cv2.waitKey(0)
