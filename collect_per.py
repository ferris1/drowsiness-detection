#-*- coding: UTF-8 -*-
# write by feng
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import pandas as pd
import dlib
import cv2
import os
import csv
import math
import common
import time 

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str, default="",
    help="path to input video file")
args = vars(ap.parse_args())

print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")
(lStart, lEnd) = (42,48)    #eye dex
(rStart, rEnd) = (36,42)
(mstart, mend) = (60,68)
class_num = 2
print("[INFO] starting video stream thread...")

#获取摄像头
vs = cv2.VideoCapture(0)
time.sleep(1.0)

if not os.path.exists('train'):
    os.mkdir('train')
train_csv = open(common.data_file,'ab+')

#写头
headers=[]
for i in range(common.VECTOR_SIZE/4):
    headers.append('eye{}'.format(i))
    headers.append('mouth{}'.format(i))
    headers.append('turn{}'.format(i))
    headers.append('nod{}'.format(i))
headers.append('label')
print(headers)
csvwriter = csv.writer(train_csv, delimiter=',')
line=train_csv.readline() 
if not line:  
    csvwriter.writerow(headers)

count=[0,0]
flag = -1
vector = []
eye_per = 0.0
normal = [[0.0,0.0,0.0]]
normal_frame = 30
count_frame = 0
queue_length = 0
dataframe = pd.read_csv(common.normal_file,header=0)
print(dataframe)
normal = dataframe.values

while True:
    if not vs.isOpened():
        break
    
    ret,frame = vs.read()
    key = cv2.waitKey(1)
    cv2.putText(frame, "press a,s to collective data,e to pause", (10,10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)    
    if key & 0xFF == ord("a"):
        flag = 0
	queue_length = 0
        vector = []
    elif key & 0xFF == ord("s"):
        flag = 1
	queue_length = 0
        vector = []
    elif key & 0xFF == ord("e"):
        flag = -1
    if key & 0xFF == ord("q"):
        break
    if(flag > 0 and  math.fabs(normal[0][0]-0.0)<0.0001):
        break

    frame = imutils.resize(frame, width=common.Frame_width,height=common.Frame_hight)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    for rect in rects:

        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        Mouth = shape[mstart:mend]
        turn_node = [shape[1],shape[30],shape[15]]
        face = [shape[30],shape[8]]

        leftEAR = common.eye_aspect_ratio(leftEye)
        rightEAR = common.eye_aspect_ratio(rightEye)
        nod = common.nod_radio(face,turn_node)
        ear = (leftEAR + rightEAR) / 2.0 
        turn = common.turning_head(turn_node)
        mouth = common.mouth_aspect_ratio(Mouth)

        eye_per = ear / normal[0][0]
        nod_per = nod / normal[0][1]
        mouth_per = mouth / normal[0][2]

        if flag >= 0:
            queue_length,vector =common.queue_in(vector,queue_length,eye_per,turn,mouth_per,nod_per)
            
        if queue_length  >= common.VECTOR_SIZE and flag >=0:
            temp = vector[:]
            print(vector)
            temp.append(flag)
            count[flag] = count[flag]+1
            csvwriter.writerow(temp)

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        (x, y ,w , h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

        for i,(x,y) in enumerate(shape):
            cv2.circle(frame,(x,y),2,(255,0,0),-1)
            cv2.putText(frame, "{}".format(i), (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 100, 125), 2)
        cv2.putText(frame, "0:fatigue image: {}".format(count[0]), (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(frame, "1:fatigue image: {}".format(count[1]), (10, 80),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, "nod: {:.3f}".format(nod), (common.Frame_width-150,120),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, "eye: {:.3f}".format(ear), (common.Frame_width-150,30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, "turn: {:.3f}".format(turn), (common.Frame_width-150,60),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, "mouth: {:.3f}".format(mouth), (common.Frame_width-150,90),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        cv2.putText(frame, "eye_per: {:.3f}".format(eye_per), (common.Frame_width-150,150),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    cv2.imshow("Frame", frame)
    #if out.isOpened():
    #    out.write(frame)

train_csv.close()
cv2.destroyAllWindows()
vs.release()
