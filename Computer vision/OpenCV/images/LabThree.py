# -*- coding: utf-8 -*-
"""
Created on Wed May 24 12:12:10 2017

@author: Anu
"""
import numpy as np
import cv2
from matplotlib import pyplot as plt
def drawMatches(img1, kp1, img2, kp2, matches):
    
    row1 = img1.shape[0]
    col1 = img1.shape[1]
    row2 = img2.shape[0]
    col2 = img2.shape[1]

    out = np.zeros((max([row1,row2]),col1+col2,3), dtype='uint8')
    # first image 
    out[:row1,:col1] = np.dstack([img1, img1, img1])
    # next image
    out[:row2,col1:] = np.dstack([img2, img2, img2])

    for mat in matches:
        img1_x = mat.queryIdx
        img2_x = mat.trainIdx
        (x1,y1) = kp1[img1_x].pt
        (x2,y2) = kp2[img2_x].pt

        cv2.circle(out,(int(x1),int(y1)),2,(0,255,0),1)   
        cv2.circle(out,(int(x2)+col1,int(y2)),2,(0,255,0),1)
        cv2.line(out,(int(x1),int(y1)),(int(x2)+col1,int(y2)),(255,0,0),1)
    cv2.imshow('Matched Features', out)
    cv2.waitKey(0)
    cv2.destroyWindow('Matched Features')
    return out

MIN_MATCH_COUNT = 10

img2 = cv2.imread('5.jpg',0) # SourceImage
img1 = cv2.imread('as.png',0) # templateImage
sift = cv2.SIFT()
kp1, des1 = sift.detectAndCompute(img1,None)
print(len(des1))
kp2, des2 = sift.detectAndCompute(img2,None)
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

matchesfound = flann.knnMatch(des1,des2,k=2)
matchesMask = [[0,0] for i in xrange(len(matchesfound))]
goodmatch = []
for m,n in matchesfound:
    if m.distance < 0.9*n.distance:
        goodmatch.append(m)
        
if len(goodmatch)>MIN_MATCH_COUNT:
    src_points = np.float32([ kp1[m.queryIdx].pt for m in goodmatch ]).reshape(-1,1,2)
    dst_points = np.float32([ kp2[m.trainIdx].pt for m in goodmatch ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_points,dst_points,cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()
    h,w = img1.shape
    points = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    distance = cv2.perspectiveTransform(points,M)
    cv2.polylines(img2,[np.int32(distance)],True,255,3,cv2.CV_AA)
else:
    print "Not enough matches are found - %d/%d" % (len(goodmatch),MIN_MATCH_COUNT)
    matchesMask = None
    
draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

finalimg = drawMatches(img1,kp1,img2,kp2,goodmatch)

plt.imshow(finalimg, 'gray'),plt.show()