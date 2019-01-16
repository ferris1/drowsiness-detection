
import imutils
from imutils.video import FileVideoStream
from imutils.video import VideoStream
import cv2
import argparse
import time 
import common 
ap = argparse.ArgumentParser()
ap.add_argument("-n", "--name", type=str, default="",
    help="video name")
args = vars(ap.parse_args())

vs = cv2.VideoCapture(0)
time.sleep(1.0) #wait camera open

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('{}.avi'.format(args["name"]),fourcc,20.0,(common.Frame_width,common.Frame_hight))
tot = 0
flag = False 
while True:
    if not vs.isOpened():
        break
    ret,frame = vs.read()
    frame = imutils.resize(frame, width=common.Frame_width,height=common.Frame_hight)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("s"):
        flag = True
    if key == ord("q"):
        break
    if flag:
        tot += 1
        print(tot)
    if out.isOpened():
#        print("yes")
        out.write(frame)
    else:
        print("writer close")

out.release()
cv2.destroyAllWindows()
vs.release()


