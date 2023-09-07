from Control import *
from Servo import *
servo=Servo()
#I=90 M=140 E=90
#servo.setServoAngle(31,0)
angle=90
I=[15,12,8,16,19,22]
M=[14,11,9,17,20,23]
E=[13,10,31,18,21,27]
for i in I:
	servo.setServoAngle(i,90)
	
#leg1
servo.setServoAngle(15,angle)
#servo.setServoAngle(14,90)
#servo.setServoAngle(13,90)

#leg2
#servo.setServoAngle(12,angle)
#servo.setServoAngle(11,90)
#servo.setServoAngle(10,90)

#leg3
#servo.setServoAngle(9,angle)
#servo.setServoAngle(8,90)
#servo.setServoAngle(31,90)

#leg6
#servo.setServoAngle(16,angle)
#servo.setServoAngle(17,90)
#servo.setServoAngle(18,90)

#leg5
#servo.setServoAngle(19,angle)
#servo.setServoAngle(20,90)
#servo.setServoAngle(21,90)

#leg4
#servo.setServoAngle(22,angle)
#servo.setServoAngle(23,90)
#servo.setServoAngle(27,90)
