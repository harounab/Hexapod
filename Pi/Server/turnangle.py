import time
import random
from Control import *
c=Control()
c.servo.setServoAngle(1,35)
data=['CMD_MOVE', '1', '30', '0', '8', '-8']
c.run(data)
c.run(data)
c.run(data)
c.run(data)

