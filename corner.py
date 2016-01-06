import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')
import cv2

import numpy as np
import matplotlib.pyplot as plt

def gauss_kernels(size,sigma=1):
    if size<3:
        size = 3
    m = size/2
    x, y = np.mgrid[-m:m+1, -m:m+1]
    kernel = np.exp(-(x*x + y*y)/(2*sigma*sigma))
    kernel_sum = kernel.sum()
    if not sum==0:
        kernel = kernel/kernel_sum
    return kernel

def filter3by3(img, ker):
    h = img.shape[0]
    w = img.shape[1]
    new=np.zeros((h,w))
    tmp= np.zeros(9)
    for i in range(1,h-1):
        for j in range(1,w-1):
            for k in range(3):
                for l in range(3):
                    tmp[k*3+l]=img[i-1+k,j-1+l]*ker[k*3+l]
            new[i,j]= sum(tmp)
    return new

def filter3(img, ker):
    h = img.shape[0]
    w = img.shape[1]
    new=np.zeros((h,w))
    tmp= np.zeros(9)
    for i in range(1,h-1):
        for j in range(1,w-1):
            for k in range(3):
                for l in range(3):
                    tmp[k*3+l]=img[i-1+k,j-1+l]*ker[k*3+l]
            new[i,j]= np.absolute(sum(tmp))
    max=np.amax(new)
    ratio = float(255)/max
    for i in range(1,h-1):
        for j in range(1,w-1):
            new[i,j]=new[i,j]*ratio
    return new

def plot(img,result,n):
    fig = plt.figure()
    plt.gray()
    plt.imshow(img)
    plt.axis('off')
    h = img.shape[0]
    w = img.shape[1]
    for i in range(1,h-1):
        for j in range(1,w-1):
            if result[i,j]==1:
                plt.plot(j-1.5,i-1.5,',',color = 'r')
                plt.plot(j-1.5,i-0.5,',',color = 'r')
                plt.plot(j-1.5,i+0.5,',',color = 'r')
                plt.plot(j-0.5,i-1.5,',',color = 'r')
                plt.plot(j-0.5,i-0.5,',',color = 'r')
                plt.plot(j-0.5,i+0.5,',',color = 'r')
                plt.plot(j+0.5,i-1.5,',',color = 'r')
                plt.plot(j+.5,i-0.5,',',color = 'r')
                plt.plot(j+.5,i+.5,',',color = 'r')
    plt.savefig(str(n+1)+'.jpg')
    plt.close(fig)


L = ["building1.png","building2.png","checker.jpg","flower.jpg"]
SIZE = 3
K = 0.06

for n in range(len(L)):
    img = cv2.imread(L[n],cv2.IMREAD_GRAYSCALE)
    h = img.shape[0]
    w = img.shape[1]
    SobelY = [1,2,1,0,0,0,-1,-2,-1]
    SobelX = [-1,0,1,-2,0,2,-1,0,1]
    gx = filter3by3(img,SobelX)
    gy = filter3by3(img,SobelY)
    Ixx=gx*gx
    Ixy=gx*gy
    Iyy=gy*gy
    GKernel= gauss_kernels(SIZE,1).reshape((9))
    Wxx=filter3by3(Ixx,GKernel)
    Wxy=filter3by3(Ixy,GKernel)
    Wyy=filter3by3(Iyy,GKernel)
    R = np.zeros((h,w))
    Rmax = 0
    for i in range(h):
        for j in range(w):
            W = [[Wxx[i,j],Wxy[i,j]],[Wxy[i,j],Wyy[i,j]]]
            detW = np.linalg.det(W)
            traceW=np.trace(W)
            R[i,j]=detW-0.06*traceW*traceW
            if R[i,j]>Rmax:
                Rmax=R[i,j]
    result=np.zeros((h,w))
    count  = 0
    for i in range(1,h-1):
        for j in range(1,w-1):
            if R[i,j] >= 0.1*Rmax and R[i,j] > R[i-1,j-1] and R[i,j] > R[i-1,j] and R[i,j] > R[i-1,j+1] and R[i,j] > R[i,j-1] and R[i,j] > R[i,j+1] and R[i,j] > R[i+1,j-1] and R[i,j] > R[i+1,j] and R[i,j] > R[i+1,j+1]:
                result[i,j]=1
                count=count+1

    plot(img,result,n)


