import cv2
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)
kernel = np.ones((5,5),np.uint8)
draw_area = None
x1,y1=0,0

while(1):
    x, frame = cap.read()
    frame = cv2.flip(frame,1)
    if draw_area is None:
        draw_area = np.zeros_like(frame)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    low = (158,120,125)
    up = (179,255,255)
    mask = cv2.inRange(hsv,low,up)
    mask = cv2.erode(mask,kernel,iterations = 1)
    mask = cv2.dilate(mask,kernel,iterations = 1)
    cv2.imshow("noise_removed",cv2.resize(mask,None,fx=0.6,fy=0.6))
    contours,hierarchy=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        cv2.drawContours(frame, [c], -1, (0,255,0), 2)
    if contours:
        c = max(contours,key=cv2.contourArea) 
        x2,y2,w,h = cv2.boundingRect(c)
        cv2.rectangle(frame,(x2,y2),(x2+w,y2+h),(0,255,0),4)
        if x1 == 0 and y1 == 0:
            x1,y1= x2,y2     
        else:
            draw_area = cv2.line(draw_area,(x1,y1),(x2,y2), [255,0,0], 4)
        x1,y1= x2,y2
    else:
        x1,y1 =0,0
    frame = cv2.add(frame,draw_area)
    stacked = np.hstack((draw_area,frame))
    cv2.imshow('Finger Paint',cv2.resize(stacked,None,fx=0.6,fy=0.6))

    k = cv2.waitKey(1)
    if k == 27:
        break
    if k == ord('c'):
        draw_area = None
cv2.destroyAllWindows()
cap.release()
