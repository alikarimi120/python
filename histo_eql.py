
import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')
import cv2


img= cv2.imread("bee_hue.png",cv2.IMREAD_GRAYSCALE)
h = img.shape[0]
w = img.shape[1]

scol = {}
dcol={}
Ps={}
Cs={}
for m in range(h):
    for n in xrange(w):
        scol[img[m,n]]=scol.setdefault(img[m,n],0)+1

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
for m in xrange(h):
    for n in xrange(w):
        v=img[m,n]
        img[m,n]= int(Cs[v]/p)

cv2.imwrite("new1.jpg",img)
