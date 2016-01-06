import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')
import cv2

import numpy as np


L = ["test1.jpg","test2.jpg","test3.jpg"]

for n in range(len(L)):
    img = cv2.imread(L[n],cv2.IMREAD_GRAYSCALE)
    h = img.shape[0]
    w = img.shape[1]
    SobelX = [1,2,1,0,0,0,-1,-2,-1]
    SobelY = [-1,0,1,-2,0,2,-1,0,1]
    PrewittX= [1,1,1,0,0,0,-1,-1,-1]
    PrewittY = [-1,0,1,-1,0,1,-1,0,1]
    tmpX=np.zeros(9)
    tmpY=np.zeros(9)
 
    new=np.zeros((h,w))
    new0=np.zeros((h,w))
    new1=np.zeros((h,w))
    new2=np.zeros((h,w))
    new3=np.zeros((h,w))
    for i in range(1,h-1):
        for j in range(1,w-1):
            for k in range(3):
                for l in range(3):
                    tmpX[k*3+l]=img[i-1+k,j-1+l]*SobelX[k*3+l]
                    tmpY[k*3+l]=img[i-1+k,j-1+l]*SobelY[k*3+l]
            new[i,j]=np.sqrt(np.power(sum(tmpX),2)+np.power(sum(tmpY),2))
    maxV=np.amax(new)
    ratio= float(255)/maxV

    for i in range(1,h-1):
        for j in range(1,w-1):
            new[i,j]=new[i,j]*ratio
            v = new[i,j]
            new0[i,j]=255-v
            if new0[i,j]<20:
                new1[i,j]=0
            else:
                new1[i,j]=new0[i,j]
            if new0[i,j]<30:
                new2[i,j]=0
            else:
                new2[i,j]=new0[i,j]

    for i in range(1,h-1):
        for j in range(1,w-1):
            v = new[i,j]
            if ((v>=new[i,j-1] and v>=new[i,j+1]) or (v>=new[i-1,j] and v>=new[i+1,j])):
                new3[i,j]=v
            else:
                new3[i,j]=0
            new3[i,j]=255-new3[i,j]


    cv2.imwrite(str(n+1)+"-Sobel-reversed.jpg",new0)
    cv2.imwrite(str(n+1)+"-Sobel-threshold-20-reversed.jpg",new1)
    cv2.imwrite(str(n+1)+"-Sobel-threshold-30-reversed.jpg",new2)
    cv2.imwrite(str(n+1)+"-Sobel-threshold-30-reversed-thinned.jpg",new3)

    new4=np.zeros((h,w))
    new5=np.zeros((h,w))
    new6=np.zeros((h,w))
    new7=np.zeros((h,w))
    new8=np.zeros((h,w))
    for i in range(1,h-1):
        for j in range(1,w-1):
            for k in range(3):
                for l in range(3):
                    tmpX[k*3+l]=img[i-1+k,j-1+l]*PrewittX[k*3+l]
                    tmpY[k*3+l]=img[i-1+k,j-1+l]*PrewittY[k*3+l]
            new4[i,j]=np.sqrt(np.power(sum(tmpX),2)+np.power(sum(tmpY),2))

    maxV=np.amax(new4)
    ratio= float(255)/maxV

    for i in range(1,h-1):
        for j in range(1,w-1):
            new4[i,j]=new4[i,j]*ratio
            v = new4[i,j]
            new5[i,j]=255-v
            if new5[i,j]<20:
                new6[i,j]=0
            else:
                new6[i,j]=new5[i,j]
            if new5[i,j]<30:
                new7[i,j]=0
            else:
                new7[i,j]=new5[i,j]
    for i in range(1,h-1):
        for j in range(1,w-1):
            v = new4[i,j]
            if ((v>=new4[i,j-1] and v>=new4[i,j+1]) or (v>=new4[i-1,j] and v>=new4[i+1,j])):
                new8[i,j]=v
            else:
                new8[i,j]=0
            new8[i,j]=255-new8[i,j]


    cv2.imwrite(str(n+1)+"-Prewitt-reversed.jpg",new5)
    cv2.imwrite(str(n+1)+"-Prewitt-threshold-20-reversed.jpg",new6)
    cv2.imwrite(str(n+1)+"-Prewitt-threshold-30-reversed.jpg",new7)
    cv2.imwrite(str(n+1)+"-Prewitt-threshold-30-reversed-thinned.jpg",new8)
    