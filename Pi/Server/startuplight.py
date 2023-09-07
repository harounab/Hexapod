import time 
from Led import *
led=Led()
led.rainbow(led.strip)
for j in range(20):
	for i in range(7):
		led.strip.setPixelColor(i, Color(0,255-j*12,0))
		led.strip.show()
		time.sleep(1/100)
for i in range(7):		
	led.strip.setPixelColor(i, Color(0,0,0))
	led.strip.show()
