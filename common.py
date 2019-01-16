#-*- coding: UTF-8 -*-
# write by feng
from scipy.spatial import distance as dist  # 这个input就是你电脑上要装才能运行的模块
import pyaudio
import wave
import cv2
import pandas as pd
beishu = 1000
VECTOR_SIZE = 80
normal_file = "./train/normal.csv"
model_file = "./train/model_per_80.h5"
data_file = "./train/train_per_80.csv"
Maxsize = 60
wav_file = "./data/dive.wav"
camera = 0

Frame_hight = 800 #int(vs.get(4))
Frame_width = 1024#int(vs.get(3))
ARXES = int(Frame_hight * (6.0/7))
EYE_OPEN = 0.30
Per_width = (Frame_width) / Maxsize
Per_hight = 1000
Queue = []

def get_base():
    dataframe = pd.read_csv(normal_file,header=0)
    normal = dataframe.values
    print(normal)
    return normal
def play_alarm():
    chunk = 1024  
    f = wave.open(wav_file,"rb")  
    p = pyaudio.PyAudio()
    stream = p.open(format = p.get_format_from_width(f.getsampwidth()),  
        channels = f.getnchannels(),  
        rate = f.getframerate(),  
        output = True)  
    data = f.readframes(chunk)  
    while data:
        stream.write(data)
        data = f.readframes(chunk)
    stream.stop_stream()  
    stream.close()  
    p.terminate()
def draw_sign(img):  #这个是画波形图的
    pre_x = 100
    pre_y = 100
    cv2.line(img,(0,ARXES),(Frame_width,ARXES),(0,0,255),2)
    for i,d in enumerate(Queue):
        x = int((i+1)*Per_width)
        if i == 0:
            y = int(ARXES)
            cv2.circle(img,(x,y),4,(0,255,0),-1)
        else:
            y = int(ARXES + ( d - EYE_OPEN)*Per_hight)
            cv2.circle(img,(x,y),4,(0,255,0),-1)
            cv2.line(img,(pre_x,pre_y),(x,y),(255,0,0),2)
        #print('x={},y={},i={},d={},len={}'.format(x,y,i,d,len(Queue)))
        pre_x = x
        pre_y = y
    return img

def save_date(data):
    Queue.append(data)
    if len(Queue) > Maxsize:
        Queue.pop(0)


def queue_in(queue, queue_length,eye, turn, mouth,nod):
    eye = int(eye*beishu)
    head = int(turn*beishu)
    mouth = int(mouth*beishu)
    nod = int(nod *beishu)
    queue += [eye,head,mouth,nod]
    #print(queue)
    queue_length+=4
    while queue_length > VECTOR_SIZE:
        queue.pop(0)
	queue_length-=1
    return queue_length,queue

def nod_radio(faces,turn_node):
    A = dist.euclidean(faces[0],faces[1])
    B = dist.euclidean(turn_node[0],turn_node[2])
    return A/B

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def turning_head(turn_node):
    Left = dist.euclidean(turn_node[0],turn_node[1])
    Righ = dist.euclidean(turn_node[1],turn_node[2])
    value = Left / Righ
    return value

def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[1],mouth[7])
    B = dist.euclidean(mouth[2],mouth[6])
    C = dist.euclidean(mouth[3],mouth[5])
    D = dist.euclidean(mouth[0],mouth[4])
    mouth_dis = (A+B+C)/ (3.0 * D)
    return mouth_dis

