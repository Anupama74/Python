# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2
plt.rcParams ['image.cmap'] = 'gray'

#nuskaito paveiksliuka
img = plt.imread('building.jpg')
plt.imshow(img)

#padaro nespalvota
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#plt.figure()
#plt.imshow(gray)

#suranda kampus

edges = cv2.Canny(gray,
                  100,
                  200,
                  apertureSize=3)
#plt.figure()
#plt.imshow(edges)


#nubreze grafika tasko
theta= np.linspace(0, 2*np.pi, 10)
x0 = 8
y0 = 6
r = x0 * np.cos(theta) + y0 * np.sin(theta)
#plt.figure()
#plt.plot(theta, r)

x1 = 4
y1 = 9
rl = x1 * np.cos(theta) + y1 * np.sin(theta)
plt.plot(theta, rl)

x2 = 12
y2 = 3
rl = x2 * np.cos(theta) + y2 * np.sin(theta)
#plt.plot(theta, rl)

#houhg transformacija

lines = cv2.HoughLines(edges,
                       1,
                       np.pi/180,
                       150) #150 kiek kreiciu susikerta i viena taska, skaicius
for r, theta in lines.squeeze():
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*r
    y0 = b*r
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*a)
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*a)
    cv2.line(img,
             (x1, y1),
             (x2, y2),
             (0, 0, 255),
             2)
    
#ieskom apskritimo

img = plt.imread('speed_limit.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.figure()
plt.imshow(gray)

edges = cv2.Canny(gray,
                  100,
                  200,
                  apertureSize=3)

circles = cv2.HoughCircles(edges,
                           cv2.HOUGH_GRADIENT,
                           1,
                           minDist = 40,
                           param1 = 50,
                           param2 = 27,
                           minRadius = 50,
                           maxRadius= 100)

for x, y, r in circles.squeeze():
    print x, y, r
    cv2.circle(gray, (x,y), r, (0, 0, 255), 2)
    
    
   
    
    
