from pynq_dpu import DpuOverlay
overlay = DpuOverlay("dpu.bit")
import os
import time
import numpy as np
import cv2
import time
from itertools import count
from multiprocessing import Process
print("aaa")
WIDTH  = 224 
HEIGHT = 224 

#normalization factor
NORM_FACTOR = 127.5

#number of classes
NUM_CLASSES = 5

# names of classes
CLASS_NAMES =("background",
		"end",
		"floor",
		"trash",
		"wall")

BATCH_SIZE = 8
EPOCHS = 300




#######################################################################################################

# colors for segmented classes
colorB = [ 0 , 0 , 255 , 116, 116]
colorG = [ 0, 255, 0, 47 , 47 ]
colorR = [ 255 , 0 , 0 , 97 , 97 ]
CLASS_COLOR = list()
for i in range(0, 5):
    CLASS_COLOR.append([colorR[i], colorG[i], colorB[i]])
COLORS = np.array(CLASS_COLOR, dtype="float32")
#######################################################################################################
#def visualize_legend():
    # initialize the legend visualization
#    legend = np.zeros( ((NUM_CLASSES * 25) + 25, 300, 3), dtype="uint8")
    # loop over the class names + colors
#    for (i, (className, color)) in enumerate(zip(CLASS_NAMES, COLORS)):
#        # draw the class name + color on the legend
#        color = [int(c) for c in color]
#        cv2.putText(legend, className, (5, (i * 25) + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
#        cv2.rectangle(legend, (100, (i * 25)), (300, (i * 25) + 25), tuple(color), -1)
#        B,G,R = cv2.split(legend)
#        legend_rgb = cv2.merge((R,G,B))
#    cv2.imshow("BGR Legend", legend)
#    cv2.imshow("RGB Legend", legend_rgb)
#    cv2.imwrite("legend_rgb.png", legend_rgb)
#    cv2.imwrite("legend_bgrb.png", legend)
#    cv2.waitKey(0)

#######################################################################################################
overlay.load_model("fcn8.xmodel")
dpu = overlay.runner
inputTensors = dpu.get_input_tensors()
outputTensors = dpu.get_output_tensors()

shapeIn = tuple(inputTensors[0].dims)
shapeOut = tuple(outputTensors[1].dims)
outputSize = int(outputTensors[1].get_data_size() / shapeIn[0])

print("bb")
def give_color_to_seg_img(seg,n_classes):
    if len(seg.shape)==3:
        seg = seg[:,:,0]
    seg_img = np.zeros( (seg.shape[0],seg.shape[1],3) ).astype('float')
    #colors = sns.color_palette("hls", n_classes) #DB
    colors = COLORS #DB
    for c in range(n_classes):
        segc = (seg == c)
        seg_img[:,:,0] += (segc*( colors[c][0]))
        seg_img[:,:,1] += (segc*( colors[c][1]))
        seg_img[:,:,2] += (segc*( colors[c][2]))

    return(seg_img)
def detect_img(img2):
    img = cv2.resize(img2, (WIDTH, HEIGHT))
    img = img.astype(np.float32)
    img = img/NORM_FACTOR - 1.0
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    norm_test_img=np.array(img)
    norm_test_img = np.reshape(norm_test_img, (1, 224, 224, 3))
    output_data = [np.empty(shapeOut, dtype=np.float32, order="C")]
    input_data = [np.empty(shapeIn, dtype=np.float32, order="C")]
    image = input_data[0]
    image[0,...] = norm_test_img.reshape(shapeIn[1:])
    job_id = dpu.execute_async(input_data, output_data)
    dpu.wait(job_id)
    y_pred1=np.array(output_data)
    y_pred1_i = np.argmax(y_pred1, axis=4)
    y_pred1_i=np.reshape(y_pred1_i,(224,224))
    final_img=give_color_to_seg_img(y_pred1_i,5)
    return final_img