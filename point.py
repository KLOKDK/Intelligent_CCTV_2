import numpy as np
import cv2
import math
import matplotlib.pyplot as plt

def find_position(points):
    sum_xy = np.sum(points, axis=1)
    diff_xy = np.diff(points, axis=1)
    
    topLeft = points[sum_xy.argmin()]
    bottomRight = points[sum_xy.argmax()]
    topRight = points[diff_xy.argmin()]
    bottomLeft = points[diff_xy.argmax()]
    
    return topLeft, topRight, bottomRight, bottomLeft

def output_size(topLeft, topRight, bottomRight, bottomLeft):
    w1, w2 = bottomRight[0] - bottomLeft[0], topRight[0] - topLeft[0]
    h1, h2 = bottomRight[1] - topRight[1], bottomLeft[1] - topLeft[1] 
    
    return int(max(w1, w2)), int(max(h1, h2))

def on_mouse(event, x, y, flags, param):
    global count, src, dst, width, height
    
    if event == cv2.EVENT_LBUTTONDOWN:       
        cv2.circle(draw_points, (x, y), 10, (0,0,255), 3)
        cv2.imshow('set points', draw_points)

        points[count] = (x, y)
        count += 1
        if count == 4:
            topL, topR, bottomR, bottomL = find_position(points)
            
            src = np.float32([topL, topR, bottomR, bottomL])
            cv2.polylines(draw_points, [np.array(src,np.int32)], True, (0,255,0), 5)
            
            width, height = output_size(topL, topR, bottomR, bottomL)

            dst = np.float32([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]])
            cv2.destroyAllWindows()

count = 0
points = np.zeros((4, 2))
draw_points = img.copy()