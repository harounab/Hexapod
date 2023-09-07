# -*- coding: utf-8 -*-
import io
import time
import fcntl
import socket
import struct
from picamera2 import Picamera2,Preview
from picamera2.encoders import JpegEncoder
from picamera2.outputs import FileOutput
from picamera2.encoders import Quality
from threading import Condition
import threading
from Led import *
from Servo import *
from Thread import *
from Buzzer import *
from Control import *
from ADC import *
from Ultrasonic import *
from Command import COMMAND as cmd

class StreamingOutput(io.BufferedIOBase):
    def __init__(self):
        self.frame = None
        self.condition = Condition()

    def write(self, buf):
        with self.condition:
            self.frame = buf
            self.condition.notify_all()

class Server:
	def __init__(self):
		self.tcp_flag=False
		self.led=Led()
		self.adc=ADC()
		self.servo=Servo()
		self.buzzer=Buzzer()
		self.control=Control()
		self.sonic=Ultrasonic()
		self.control.Thread_conditiona.start()
	def forward(self):
		self.control.run(['CMD_MOVE', '2', '35', '0', '10', '10'])

servo=Server()
servo.forward()
