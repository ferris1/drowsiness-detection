#-*- coding: UTF-8 -*-
# write by feng
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import os
import csv
import math
import common

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
normal_file = open(common.normal_file,'w')
headers = ['normal_eye','normal_nod','normal_mouth']
csvwriter = csv.writer(normal_file, delimiter=',')
csvwriter.writerow(headers)

flag = -1
normal = [0.0,0.0,0.0]
normal_num = 30
total =  [0.0,0.0,0.0]
total_num = 0
while True:
    if not vs.isOpened():
        break
    key = cv2.waitKey(1)
    ret,frame = vs.read()
    if flag == -1:
        cv2.putText(frame, "please keep your face normally. a to start", (10,10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2) 
	if key & 0xFF == ord("a"):
            flag = 1
    elif flag == 0:
        cv2.putText(frame, "please open your mouth to max . s to start", (10,10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2) 
        if key & 0xFF == ord("s"):
            flag = 2

    if key & 0xFF == ord("q"):
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

        leftEAR =common.eye_aspect_ratio(leftEye)
        rightEAR = common.eye_aspect_ratio(rightEye)
        nod = common.nod_radio(face,turn_node)
        ear = (leftEAR + rightEAR) / 2.0 
        head =common.turning_head(turn_node)
        mouth =common.mouth_aspect_ratio(Mouth)

        if flag ==1:
            total[0] += ear
            total[1] += nod
            total_num += 1
            print(total[0])
            if(total_num >= normal_num):
                normal[0] = total[0] / total_num
                normal[1] = total[1] / total_num
                flag = 0
                total_num = 0
        elif flag == 2:
            total[2] += mouth 
            total_num += 1
            print(total[2])
            if(total_num >= normal_num):
                normal[2] = total[2] / total_num
                flag = -2
                total_num = 0

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
        cv2.putText(frame, "nod: {:.3f}".format(nod), (common.Frame_width-150,120),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, "eye: {:.3f}".format(ear), (common.Frame_width-150,30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, "head: {:.3f}".format(head), (common.Frame_width-150,60),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, "mouth: {:.3f}".format(mouth), (common.Frame_width-150,90),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    cv2.imshow("Frame", frame)
    #if out.isOpened():
    #    out.write(frame)

print(normal)
csvwriter.writerow(normal)
normal_file.close()
cv2.destroyAllWindows()
vs.release()
