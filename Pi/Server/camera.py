from picamera2 import Picamera2
picam2 = Picamera2()
for i in range(74,300):
	picam2.start_and_capture_file("image{}.jpg".format(i))
