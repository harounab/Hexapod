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
from tflite_runtime.interpreter import Interpreter
import cv2
import numpy as np
import time
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
model_path = "model.tflite"
interpreter = Interpreter(model_path)
#print("Model Loaded Successfully.")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']
print( " input shape " , input_shape)
def detect(image):
    interpreter.set_tensor(input_details[0]['index'],image.reshape(1,224,224,3))

    interpreter.invoke()

    # The function get_tensor() returns a copy of the tensor data.
    # Use tensor() in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predi=np.argmax(output_data,axis=3)
    #print(predi)
    final_img= give_color_to_seg_img(predi.reshape(224,224),5)
    return final_img
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
        self.action_flag=0
        self.control.Thread_conditiona.start()
    def get_interface_ip(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        return socket.inet_ntoa(fcntl.ioctl(s.fileno(),
                                            0x8915,
                                            struct.pack('256s',b'wlan0'[:15])
                                            )[20:24])
    def turn_on_server(self):
        #ip adress
        HOST=self.get_interface_ip()
        #Port 8002 for video transmission
        self.server_socket = socket.socket()
        self.server_socket.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEPORT,1)
        self.server_socket.bind((HOST, 8002))              
        self.server_socket.listen(1)
        
        #Port 5002 is used for instruction sending and receiving
        self.server_socket1 = socket.socket()
        self.server_socket1.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEPORT,1)
        self.server_socket1.bind((HOST, 5002))
        self.server_socket1.listen(1)
        print('Server address: '+HOST)
        
    def turn_off_server(self):
        try:
            self.connection.close()
            self.connection1.close()
        except :
            print ('\n'+"No client connection")
    
    def reset_server(self):
        self.turn_off_server()
        self.turn_on_server()
        self.video=threading.Thread(target=self.transmission_video)
        self.instruction=threading.Thread(target=self.receive_instruction)
        self.video.start()
        self.instruction.start()
    def send_data(self,connect,data):
        try:
            connect.send(data.encode('utf-8'))
            #print("send",data)
        except Exception as e:
            print(e)
    def transmission_video(self):
        try:
            self.connection,self.client_address = self.server_socket.accept()
            self.connection=self.connection.makefile('wb')
        except:
            pass
        self.server_socket.close()
        print ("socket video connected ... ")
        camera = Picamera2()
        camera.configure(camera.create_video_configuration(main={"size": (400, 300)}))
        output = StreamingOutput()
        encoder = JpegEncoder(q=40)
        camera.start_recording(encoder, FileOutput(output),quality=Quality.LOW) 
        i=0
        while True:
            with output.condition:
                output.condition.wait()
                frame = output.frame
                image0 = cv2.imdecode(np.frombuffer(frame, dtype=np.uint8), cv2.IMREAD_COLOR)
                print(" frame shape" , image0.shape)
                #image = frame
                image = cv2.resize(image0,(224,224))
                image = np.array(image, dtype='float32')
                if i % 5 == 0:
                    final = detect(image)
                    final = final.astype(np.float32)
                    image1_norm = cv2.normalize(image.astype('float32'), None, 0.0, 1.0, cv2.NORM_MINMAX)
                    image2_norm = cv2.normalize(final.astype('float32'), None, 0.0, 1.0, cv2.NORM_MINMAX)
                    combined_image = cv2.addWeighted(image1_norm, 0.6, image2_norm, 0.4, 0)
                    combined_image = (combined_image * 255).astype('uint8')
                    cv2.imshow("image",combined_image)
                    cv2.waitKey(1000//25)
                i=i+1
                #print('final ',final.shape)
            try:                
                lenFrame = len(output.frame) 
                #print("output .length:",lenFrame)
                lengthBin = struct.pack('<I', lenFrame)
                self.connection.write(lengthBin)
                self.connection.write(frame)
            except Exception as e:
                camera.stop_recording()
                camera.close()
                print ("End transmit ... " )
                break
    def move(self,move_point):
        try:
            print("x mapping")
            x = self.map((move_point[0]-325),0,100,0,35)
            print("x mapped")
            y = self.map((635 - move_point[1]),0,100,0,35)
            print('y mapped')
            if self.action_flag == 1:
                angle = 0
                print("angle =0")
            else:
                if x!=0 or y!=0:
                    print("x or y not 0")
                    angle=math.degrees(math.atan2(x,y))

                    if angle < -90 and angle >= -180:
                        angle=angle+360
                    if angle >= -90 and angle <=90:
                        angle = self.map(angle, -90, 90, -10, 10)
                    else:
                        angle = self.map(angle, 270, 90, 10, -10)
                        print("angle mapped")
                else:
                    angle=0
            speed="10"
            print("executing")
            data = [ "CMD_MOVE",1, str(x),str(y),"10",angle  ]
            self.control.run(data)
        except: 
            print("Error")
    def map(self, value, fromLow, fromHigh, toLow, toHigh):
        return (toHigh - toLow) * (value - fromLow) / (fromHigh - fromLow) + toLow
    def adas(self):
            forward_move_point = [325, 535]
            backward_move_point =[325, 735]
            left_move_point = [225, 635]
            right_move_point =  [425, 635]
            if self.sonic.getDistance()>10:
                print(self.sonic.getDistance())
                data=['CMD_MOVE', '1', '-5', '25', '10', '0']          
            else: 
                choice = random.choi([0,1])
                if choice : #go left 
                    data=['CMD_MOVE', '1', '-25', '0', '10', '0']
                else : 
                    data=['CMD_MOVE', '1', '25', '0', '10', '0']
            if True :
                self.action_flag=1
            else:
                self.action_flag=0

    def receive_instruction(self):
        try:
            self.connection1,self.client_address1 = self.server_socket1.accept()
            print ("Client connection successful !")
        except:
            print ("Client connect failed")
        self.server_socket1.close()
        
        while True:
            try:
                allData=self.connection1.recv(1024).decode('utf-8')
            except:
                if self.tcp_flag:
                    self.reset_server()
                    break
                else:
                    break
            if allData=="" and self.tcp_flag:
                self.reset_server()
                break
            else:
                cmdArray=allData.split('\n')
                print(cmdArray)
                if cmdArray[-1] !="":
                    cmdArray==cmdArray[:-1]
            for oneCmd in cmdArray:
                data=oneCmd.split("#")
                if data==None or data[0]=='':
                    continue
                elif cmd.CMD_BUZZER in data:
                    self.buzzer.run(data[1])
                elif cmd.CMD_POWER in data:
                    try:
                        batteryVoltage=self.adc.batteryPower()
                        command=cmd.CMD_POWER+"#"+str(batteryVoltage[0])+"#"+str(batteryVoltage[1])+"\n"
                        #print(command)
                        self.send_data(self.connection1,command)
                        if batteryVoltage[0] < 5.5 or batteryVoltage[1]<6:
                         for i in range(3):
                            print('under voltage')
                            #self.buzzer.run("1")
                            #time.sleep(0.15)
                            #self.buzzer.run("0")
                            #time.sleep(0.1)
                    except:
                        pass
                elif cmd.CMD_LED in data:
                    try:
                        stop_thread(thread_led)
                    except:
                        pass
                    thread_led=threading.Thread(target=self.led.light,args=(data,))
                    thread_led.start()   
                elif cmd.CMD_LED_MOD in data:
                    try:
                        stop_thread(thread_led)
                        #print("stop,yes")
                    except:
                        #print("stop,no")
                        pass
                    thread_led=threading.Thread(target=self.led.light,args=(data,))
                    thread_led.start()
                elif cmd.CMD_SONIC in data:
                    command=cmd.CMD_SONIC+"#"+str(self.sonic.getDistance())+"\n"
                    self.send_data(self.connection1,command)
                elif cmd.CMD_HEAD in data:
                    if len(data)==3:
                        self.servo.setServoAngle(int(data[1]),int(data[2]))
                elif cmd.CMD_CAMERA in data:
                    if len(data)==3:
                        x=self.control.restriction(int(data[1]),50,180)
                        y=self.control.restriction(int(data[2]),0,180)
                        self.servo.setServoAngle(0,x)
                        self.servo.setServoAngle(1,y)
                elif cmd.CMD_RELAX in data:
                    #print(data)
                    if self.control.relax_flag==False:
                        self.control.relax(True)
                        self.control.relax_flag=True
                    else:
                        self.control.relax(False)
                        self.control.relax_flag=False
                elif cmd.CMD_SERVOPOWER in data:
                    if data[1]=="0":
                        GPIO.output(self.control.GPIO_4,True)
                    else:
                        GPIO.output(self.control.GPIO_4,False)
                    
                else:
                    self.control.order=data
                    self.control.timeout=time.time()
        try:
            stop_thread(thread_led)
        except:
            pass
        try:
            stop_thread(thread_sonic)
        except:
            pass
        print("close_recv")

if __name__ == '__main__':
    s=Server()
    s.adas()
    pass
    
