#-*- coding: UTF-8 -*-
# write by feng

from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import math
import cv2
from sklearn.externals import joblib
from threading import Thread
from keras.models import load_model
import common
import pandas as pd
#import get_normal

model = load_model(common.model_file)
#input argument 
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
# loading train mkdel

print("[INFO] starting video stream thread...")

vs = cv2.VideoCapture(common.camera)
time.sleep(1.0) #wait camera open

vector =[]
lower = [0,0,0,0]
lower1 = [0,0,0,0]
high = [0,0,0,0]  #
high1 = [0,0,0,0]
percentage = 0.45 #判断占比
ALARM_ON = False
warning_time = 30
eye_per = 0.0
flag = -1 
fatigue = 0.0
normal = common.get_base()
queue_length = 0

#save video
#fps = vs.get(cv2.CAP_PROP_FPS) #获取视频的帧率
#size = (int(vs.get(cv2.CAP_PROP_FRAME_WIDTH)),int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT)))#获取视频的大小
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter('bink_t.avi',fourcc,10.0,(common.Frame_width,common.Frame_hight))

while True:
    if not vs.isOpened():
        break
 
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

    if(math.fabs(normal[0][0]-0.0)<0.0001):
        break

    ret,frame = vs.read()
    frame = imutils.resize(frame, width=common.Frame_width,height=common.Frame_hight)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    for k,rect in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        mouth  = shape[mstart:mend]
        turn_node = [shape[1],shape[30],shape[15]]
        face = [shape[30],shape[8]]

        leftEAR = common.eye_aspect_ratio(leftEye)
        rightEAR = common.eye_aspect_ratio(rightEye)
        
        nod = common.nod_radio(face,turn_node)
        ear = (leftEAR + rightEAR) / 2.0
        turn = common.turning_head(turn_node)
        mouth_dis = common.mouth_aspect_ratio(mouth)
        
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)

        common.save_date(ear)
        frame = common.draw_sign(frame) 

        eye_per = ear / normal[0][0]
        nod_per = nod / normal[0][1]
        mouth_per = mouth_dis / normal[0][2]
        queue_length,vector = common.queue_in(vector,queue_length,eye_per,turn,mouth_per,nod_per)
        res = [[0]]
        if queue_length >= common.VECTOR_SIZE:
            #print(vector)
            input_vector = []
            input_vector.append(vector)
            inputs = np.array(input_vector)
            inputs.reshape(common.VECTOR_SIZE,1)
            res = model.predict(inputs)
            if res[0][0] < 0.5:
                lower[k] += 1
                lower1[k] += 1
            else:
                high[k] += 1
                high1[k] += 1
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        (x, y ,w , h) = face_utils.rect_to_bb(rect)
        for i,(dx,dy) in enumerate(shape):
            cv2.circle(frame,(dx,dy),2,(255,0,0),-1)


        cv2.putText(frame, "nod: {:.3f}".format(nod), (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, "eye: {:.2f}".format(ear), (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, "turn: {:.2f}".format(turn), (10, 90),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(frame, "mouth: {:.2f}".format(mouth_dis), (10, 120),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255,0), 2)
        cv2.putText(frame, "lower1: {}".format(lower1[k]), (10, 150),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        cv2.putText(frame, "high1: {}".format(high1[k]), (10, 180),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)


        if high[k]+lower[k] > warning_time:
            fatigue = float(high[k])/(high[k]+lower[k])
            if high[k] > (high[k]+lower[k])*percentage:
                #play_alarm(Wavfile)
                ALARM_ON= True
            else: 
                ALARM_ON = False
            high[k] = lower[k] = 0

        if ALARM_ON:
            color_show = (0,0,255)
        else:   
            color_show = (0,255,0)

        cv2.putText(frame, "lower: {}".format(lower[k]), (x, y-60),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        cv2.putText(frame, "high: {}".format(high[k]), (x, y-40),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        cv2.putText(frame, "fatigue: {:.2f}%".format(fatigue*100), (x, y-10),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_show, 4)    
        cv2.rectangle(frame,(x,y),(x+w,y+h),color_show,2)

    cv2.imshow("Frame", frame)
#    if out.isOpened():
#        out.write(frame)
#    else:
#        print("writer close")
#out.release()
vs.release()
cv2.destroyAllWindows()
