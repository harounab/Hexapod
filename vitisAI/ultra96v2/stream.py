import socket
import struct
import threading
import numpy as np
import multiprocessing
from PIL import Image, ImageDraw
import fcntl
import cv2
from dpuproces import *
def encode_to_4_bytes(image):
    # Convert the CV2 image to bytes using OpenCV's imencode function
    success, encoded_image = cv2.imencode('.jpg', image)

    if not success:
        # Handle the case where the encoding fails
        raise Exception("Failed to encode image to bytes.")

    # Convert the encoded image to bytes format
    image_bytes = np.array(encoded_image).tobytes()
class Client:
    def __init__(self):
        self.tcp_flag=False
        self.video_flag=True
        self.fece_recognition_flag = False
        self.image=''
        self.connection =None
        self.i=0
        self.destination_ip = '192.168.3.1'
        self.destination_connection=None
    def get_interface_ip(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        return socket.inet_ntoa(fcntl.ioctl(s.fileno(),
                                            0x8915,
                                            struct.pack('256s',b'wlan0'[:15])
                                            )[20:24])
    def turn_on_server(self):
        #ip adress
        HOST="172.20.10.7"
        #Port 8002 for video transmission
        self.server_socket_destination = socket.socket()
        self.server_socket_destination.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEPORT,1)
        self.server_socket_destination.bind((self.destination_ip , 8002))              
        self.server_socket_destination.listen(1)
    def turn_off_server(self):
        try:
            self.destination_connection.close()
        except :
            print ('\n'+"No client connection")
        
    def turn_on_client(self,ip):
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print (ip)
    def turn_off_client(self):
        try:
            self.client_socket.shutdown(2)
            self.client_socket.close()
        except Exception as e:
            print(e)
    def is_valid_image_4_bytes(self,buf): 
        bValid = True
        if buf[6:10] in (b'JFIF', b'Exif'):     
            if not buf.rstrip(b'\0\r\n').endswith(b'\xff\xd9'):
                bValid = False
        else:        
            try:  
                Image.open(io.BytesIO(buf)).verify() 
            except:  
                bValid = False
        return bValid
    def writebytes(self,result):
        frame = encode_to_4_bytes(result)
        lenFrame = len(frame) 
        #print("output .length:",lenFrame)
        lengthBin = struct.pack('<I', lenFrame)
        self.connection.write(lengthBin)
        self.connection.write(frame)
        
    def receiving_video(self,ip):
        while True:
            print("connecting source")
            self.client_socket.connect((ip, 8002))
            print("source connected")
            self.connection = self.client_socket.makefile('rb')
            if self.client_socket is not None :
                print("connected succesfully..")
                break
        #while True:
            #print("connecting dest")
            #self.destination_connection,self.client_address_destination = self.server_socket_destination.accept()
            #print("dest connected")
            #if self.destination_connection is not None:
             #   print("breaking")
             #   break
        #self.destination_connection=self.destination_connection.makefile('wb')
            
        while True:
            try:
                stream_bytes= self.connection.read(4)
                leng=struct.unpack('<L', stream_bytes[:4])
                jpg=self.connection.read(leng[0])
                if self.is_valid_image_4_bytes(jpg):
                    if self.video_flag:
                        image = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                        self.i=self.i+1
                        result = detect_img(image)
                        self.writebytes(result)
                        if self.i== 10000:
                            break
            except BaseException as e:
                print (e)
                break


ip="172.20.10.7"
c=Client()
c.turn_on_client(ip)
#c.turn_on_server()
c.receiving_video(ip)