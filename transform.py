import numpy as np
import cv2

def topdown_transform(image, pts):
    rect = pts
    
    (tl, tr, br, bl) = rect
    maxWidth = 10460
    maxHeight = 6800
      
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    
    M = cv2.getPerspectiveTransform(rect, dst)
    result = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return result

image = cv2.imread("result.jpg")

# top-left, top-right, bottom-right, bottom-left
pts = np.array([[4800.29,633.28],[7218.85,612.523],[10716.9,1422.17],[1281.44,1629.77]],dtype = "float32")

warped = topdown_transform(image, pts)

cv2.imwrite("warped.jpg",warped)