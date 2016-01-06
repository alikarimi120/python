import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')
import cv2
import numpy as np
import math

def rgb2hsv(r, g, b):
    
    r = r/255.0
    g = g/255.0
    b = b/255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx-mn
    if mx == mn:
        h = 0
    elif mx == r:
        h = 60 * ((g-b)/df % 6)
    elif mx == g:
        h = 60 * ((b-r)/df + 2)
    elif mx == b:
        h = 60 * ((r-g)/df + 4)
    if h < 0:
        h = h + 360
    if mx == 0:
        s = 0
    else:
        s = df/mx
    v = mx
    return int(h/360*255), int(s*255), int(v*255)

def hsv2bgr(h, s, v):
    
    h = float(h)/255*360
    s = float(s)/255
    v = float(v)/255
    h_lvl = int(math.floor(h / 60.0))%6
    c = v * s
    x = c * (1 - np.absolute((h/60.0)%2-1))
    m = v - c
    r, g, b = 0, 0, 0
    if h_lvl == 0: r, g, b = c, x, 0
    elif h_lvl == 1: r, g, b = x, c, 0
    elif h_lvl == 2: r, g, b = 0, c, x
    elif h_lvl == 3: r, g, b = 0, x, c
    elif h_lvl == 4: r, g, b = x, 0, c
    elif h_lvl == 5: r, g, b = c, 0, x
    r, g, b = int((r+m)* 255), int((g+m)* 255), int((b+m) * 255)
    return b, g, r

def histeq(arr):
    h=arr.shape[0]
    w=arr.shape[1]
    scol = {}
    dcol={}
    Ps={}
    Cs={}
    for m in range(h):
        for n in range(w):
            scol[arr[m,n]]=scol.setdefault(arr[m,n],0)+1
    n = h*w
    for i in range(256):
        if scol.has_key(i):
            Ps[i]=float(scol[i])/n
        else:
            Ps[i] = 0.0

    Cs[0]=Ps[0]
    for j  in range(1,256):
        Cs[j]=Cs[j-1]+Ps[j]
    p=float(1)/256
    v= 0
    for m in range(h):
        for n in range(w):
            v=arr[m,n]
            arr[m,n]= int(Cs[v]/p)
    return arr

img = cv2.imread("flower.jpg")
h = img.shape[0]
w = img.shape[1]
hsv_img = np.zeros((h,w,3))
bgr_img = np.zeros((h,w,3))
hue = np.zeros((h,w))
sat = np.zeros((h,w))
val = np.zeros((h,w))

for i in range(h):
    for j in range(w):
        tmp = np.array(rgb2hsv(img[i,j][2],img[i,j][1],img[i,j][0]))
        hsv_img[i,j] = tmp
        hue[i,j] = tmp[0]
        sat[i,j] = tmp[1]
        val[i,j] = tmp[2]

cv2.imwrite("flower_hsv.jpg",hsv_img)
cv2.imwrite("hue.jpg",hue)
cv2.imwrite("saturation.jpg",sat)
cv2.imwrite("brightness.jpg",val)

for i in range(h):
    for j in range(w):
        bgr_img[i,j]  = np.array(hsv2bgr(hsv_img[i,j][0],hsv_img[i,j][1],hsv_img[i,j][2]))

cv2.imwrite("hsv2rgb.jpg",bgr_img)

img2 = cv2.imread("bee1.png")
h2 = img2.shape[0]
w2 = img2.shape[1]
hsv_img2 = np.zeros((h2,w2,3))
bgr_img2 = np.zeros((h2,w2,3))
val2 = np.zeros((h2,w2))
val2_histeq = np.zeros((h2,w2))

for i in range(h2):
    for j in range(w2):
        tmp2 = np.array(rgb2hsv(img2[i,j][2],img2[i,j][1],img2[i,j][0]))
        hsv_img2[i,j] = tmp2
        val2[i,j] = tmp2[2]

val2_histeq = histeq(val2)
cv2.imwrite("bee_value_histeq.png",val2_histeq)

for i in range(h2):
    for j in range(w2):
        hsv_img2[i,j][2]=val2_histeq[i,j]
        bgr_img2[i,j] = np.array(hsv2bgr(hsv_img2[i,j][0],hsv_img2[i,j][1],hsv_img2[i,j][2]))
cv2.imwrite("histeq.png",bgr_img2)

