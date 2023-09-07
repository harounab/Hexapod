import numpy as np 
import cv2
import matplotlib.pyplot as plt
import os

filename= os.path.join(os.getcwd(),"eval_testing_19.png")
#detect edges
def lanes(image)
	def canny(image):
	    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	    blur = cv2.GaussianBlur(gray,(5,5),0)
	    canny = cv2.Canny(blur,0,250)
	    return canny
	#
	def make_coordinates(image,line_params):
	    slope, intercept = line_params
	    y1 = 700
	    y2= 150
	    x1= int((y1-intercept)/slope)
	    x2=int((y2-intercept)/slope)
	    return np.array([x1,y1,x2,y2])
	#get averge slope from detected lines
	def average_slope_intercept(image,lines):
	    left_fit, right_fit= [], []
	    for line in lines:
		x1, y1, x2, y2 = line.reshape(4)
		parameters= np.polyfit((x1,x2),(y1,y2),1)
		slope= parameters[0]
		intercept= parameters[1]
		print("slope",slope)
		if (np.abs(slope)<6):
		    if slope < 0:
		        left_fit.append((slope,intercept))
		    else : 
		        right_fit.append((slope,intercept))
	    left_fit_average= np.average(left_fit, axis =0)
	    print(left_fit_average,"left")
	    right_fit_average= np.average(right_fit, axis =0)
	    print(right_fit_average,"right")
	    #left_line = make_coordinates(image,left_fit_average)
	    #right_line= make_coordinates(image,right_fit_average)
	    slope_left,intercept_left = left_fit_average
	    y1_left,y2_left = 500, 200
	    x1_left,x2_left = int((y1_left-intercept_left)/slope_left),int((y2_left-intercept_left)/slope_left)
	    y1_right,y2_right = 500,200
	    
	    slope_right,intercept_right= right_fit_average
	    x1_right,x2_right = int((y1_right-intercept_right)/slope_right),int((y2_right-intercept_right)/slope_right)
	    left_line=np.array([x1_left,y1_left,x2_left,y2_left])
	    right_line=np.array([x1_right,y1_right,x2_right,y2_right])
	    middle_line=(left_line+right_line)//2
	    return np.array([right_line,left_line,middle_line])
	# mask
	def region_of_interest(image):
	    height = image.shape[0]
	    poly= np.array([[(0,100),(120,65),(149,71),(213,71),(222,222)]])
	    mask = np.zeros_like(image)
	    cv2.fillPoly(mask,poly,255)
	    masked_img= cv2.bitwise_and(mask,image)
	    return masked_img 
	#show detected lines
	def display_lines(image,lines):
	    line_image = np.zeros_like(image)
	    if lines is not None:
		for line in lines:
		    x1, y1,x2, y2=line.reshape(4)
		    cv2.line(line_image,(x1,y1),(x2,y2),(0,255,0),5)
	    return line_image    
	image =cv2.imread(filename)
	#image=cv2.resize(image,(323,621))
	gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	lane_image =np.copy(image)
	canny_img = canny(lane_image)
	cropped_img= region_of_interest(canny_img)
	lines = cv2.HoughLinesP(cropped_img,1,np.pi/180,35,np.array([]),minLineLength=20,maxLineGap=50)
	averaged_lines =average_slope_intercept(image,lines)
	print(lines)
	return lines

line_image= display_lines(image,averaged_lines)
line_image2=display_lines(image,lines)
combo_image= cv2.addWeighted(lane_image,0.8,line_image,0.3,1)
plt.imshow(combo_image)
plt.show()
# cv2.imshow("image",canny_img)
# cv2.waitKey(0)
