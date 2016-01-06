import os
os.chdir('/Users/yuanruiliu/Documents/python/bg')
import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')
import cv2
import cv2.cv as cv
import numpy as np
cap = cv2.VideoCapture("traffic.mp4")
fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)

frameCount= cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)

w = cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
h = cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
fps = int(fps)
frameCount = int (frameCount)
w=int(w)
h=int(h)
_,img = cap.read()
avgImg = np.float32(img)

for fr in range(1, frameCount):
	_,img=cap.read()
	alpha = 1/float(fr+1)
	cv2.accumulateWeighted(img,avgImg,alpha)
	normImg = cv2.convertScaleAbs(avgImg)
	cv2.imshow('img',img)
	cv2.imshow('normImg',normImg)

cv2.imwrite("bg.jpg",normImg)
cv2.waitKey(0)
cv2.destroyAllWindows()
cap.release()


