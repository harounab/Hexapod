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

import os
def encode_to_4_bytes(image):
    # Convert the CV2 image to bytes using OpenCV's imencode function
    success, encoded_image = cv2.imencode('.jpg', image)

    if not success:
        # Handle the case where the encoding fails
        raise Exception("Failed to encode image to bytes.")

    # Convert the encoded image to bytes format
    image_bytes = np.array(encoded_image).tobytes()
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
            print('connection not accepted')
            pass
        self.server_socket.close()
        print ("socket video connected ... ")
        os.system("python testtf.py")
        camera = Picamera2()
        camera.configure(camera.create_video_configuration(main={"size": (400, 300)}))
        output = StreamingOutput()
        encoder = JpegEncoder(q=50)
        camera.start_recording(encoder, FileOutput(output),quality=Quality.VERY_HIGH) 
        while True:
            with output.condition:
                output.condition.wait()
                frame = output.frame
            try:                
                lenFrame = len(output.frame) 
                #print("output .length:",lenFrame)
                lengthBin = struct.pack('<I', lenFrame)
                self.connection.write(lengthBin)
                self.connection.write(frame)
                stream_bytes= self.connection.read(4)
                leng=struct.unpack('<L', stream_bytes[:4])
                jpg=self.connection.read(leng[0])
            except Exception as e:
                camera.stop_recording()
                camera.close()
                print ("End transmit ... " )
                break

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
                
        print("close_recv")

if __name__ == '__main__':
    server=Server()
    server.turn_on_server()
    server.transmission_video()
    
