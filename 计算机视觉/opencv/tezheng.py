# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 15:45:27 2019

@author: Coffee
"""

import cv2 
import numpy as np
import matplotlib.pyplot as plt


def cv_show(name,img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
img1 = cv2.imread('img/box.png', 0)
img2 = cv2.imread('img/box_in_scene.png', 0)
cv_show('img1',img1)
cv_show('img2',img2)

#opencv 3.3之后不再免费开发SIFT
orb =  cv2.ORB()
print('start to get detect')
kp1,des1 = orb.detectAndCompute(img1, None)
kp2,des2 = orb.detectAndCompute(img2, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
print('start to  match')
matches = bf.match(des1, des2)
print('sort')
matches = sorted(matches, key=lambda x: x.distance)
img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None,flags=2)
cv_show('img3',img3)