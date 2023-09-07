from Ultrasonic import *
import time
import random
from Control import *
#Creating object 'control' of 'Control' class.
c=Control()
c.servo.setServoAngle(1,25)
sonic = Ultrasonic()
#data=['CMD_MOVE', '1', '-35', '0', '8', '0']
#c.run(data)
for i in range(100):
	angle = '0'
	if sonic.getDistance()>20:
		data=['CMD_MOVE', '1', '0', '30', '8', '0']
		c.run(data)
		data=['CMD_MOVE', '1', '-35', '0', '8', '1']
		c.run(data)
	else:
		data=['CMD_MOVE', '1', '250', '0', '10', angle] 
		c.run(data)
		c.run(data)



