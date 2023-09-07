from tflite_runtime.interpreter import Interpreter
import cv2
import numpy as np
import time
from picamera2 import Picamera2
from lanes import lanes
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
picam2.start()
from Ultrasonic import *
import time
import random
from Control import *
import os
#Creating object 'control' of 'Control' class.
c=Control()
c.servo.setServoAngle(1,35)
sonic = Ultrasonic()
#data=['CMD_MOVE', '1', '-35', '0', '8', '0']
#c.run(data)


NUM_CLASSES=5
colorB = [ 0 , 0 , 255 , 116,122,0,0,0,0]
colorG = [ 0, 255, 0, 47 ,255,0,0,0,0]
colorR = [ 255 , 0 , 0 , 97 ,112,0,0,0]
CLASS_COLOR = list()
for i in range(NUM_CLASSES):
    CLASS_COLOR.append([colorR[i], colorG[i], colorB[i]])
COLORS = np.array(CLASS_COLOR, dtype="float32")
def give_color_to_seg_img(seg,n_classes):
    if len(seg.shape)==3:
        seg = seg[:,:,0]
    seg_img = np.zeros( (seg.shape[0],seg.shape[1],3) ).astype('float')
    #colors = sns.color_palette("hls", n_classes) #DB
    colors = COLORS #DB
    for c in range(n_classes):
        segc = (seg == c)
        seg_img[:,:,0] += (segc*( colors[c][0]/255.0 ))
        seg_img[:,:,1] += (segc*( colors[c][1]/255.0 ))
        seg_img[:,:,2] += (segc*( colors[c][2]/255.0 ))
    return seg_img




# The function get_tensor() returns a copy of the tensor data.
# Use tensor() in order to get a pointer to the tensor.
def detect(image):
	interpreter.set_tensor(input_details[0]['index'],image.reshape(1,224,224,3))
	interpreter.invoke()
	output_data = interpreter.get_tensor(output_details[0]['index'])
	predi=np.argmax(output_data,axis=3)
	final_img= give_color_to_seg_img(predi.reshape(224,224),5)
	return final_img
#print(predi)
def count_matching_pixels(image, target_color):
    # Initialize counter
    matching_pixel_count = 0

    # Iterate through each pixel
    for row in image:
        for pixel in row:
            pixel_value = np.array(pixel)

            # Check if pixel matches the target color value
            if np.array_equal(pixel_value, target_color):
                matching_pixel_count += 1

    return matching_pixel_count





os.system("sudo python startuplight.py")
while True:
    im = picam2.capture_array()
    im = im[:, :, :3]
    im = im[:, :, :3]
    im = cv2.resize(im,(224,224))
    im = np.array(im, dtype='float32')

    #print(im.shape)
    stream_bytes= self.connection.read(4)
    leng=struct.unpack('<L', stream_bytes[:4])
    final=self.connection.read(leng[0])
    #print(final.shape)
    final = final.astype(np.float32)
    image1_norm = cv2.normalize(im.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    image2_norm = cv2.normalize(final.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    combined_image = cv2.addWeighted(image1_norm, 0.6, image2_norm, 0.4, 0)
    combined_image = (combined_image * 255).astype('uint8')
    angle = '0'
    #print(image2_norm)
    if sonic.getDistance()>30:
        data=['CMD_MOVE', '1', '0', '30', '10', '0']
        c.run(data)
        data=['CMD_MOVE', '1', '2', '0', '4', '-1']
        c.run(data)
    else:
        if count_matching_pixels(image2_norm,[1.,0.,0.])>20000:
            print("wall")
            #os.system("aplay /home/pi/wall.wav")
            while(sonic.getDistance()<40):
                data=['CMD_MOVE', '1', '-135', '0', '8', '8'] 
                c.run(data)
                c.run(data)
                c.run(data)
                c.run(data)
        else:
            print("obstacle")
            #os.system("aplay /home/pi/obstacle.wav")
            data=['CMD_MOVE', '1', '250', '0', '10', '0'] 
            c.run(data)
            c.run(data)
    lanes=lanes(final)
    if len(lanes) == 2:
    	if norm(lanes[0])<norm(lanes[1])):
    		data=['CMD_MOVE', '1', '-15', '30', '10', '0']
    		c.run(data)
    	else: 
    	  	data=['CMD_MOVE', '1', '15', '30', '10', '0']
    		c.run(data)
    		
    cv2.imshow("Camera", combined_image)
    cv2.waitKey(1)
