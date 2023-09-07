import cv2
import numpy 
filename="office.jpg"
image = cv2.imread(filename)
image = cv2.resize(image,(1024,682))
cv2.imwrite("office_resized.jpg", image)