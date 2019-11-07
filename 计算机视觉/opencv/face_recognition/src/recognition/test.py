# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 16:17:31 2019

@author: Coffee
"""
import face_recognition
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
import argparse
import time
import dlib
import cv2

# http://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf
def eye_aspect_ratio(eye):
	# 计算距离，竖直的
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	# 计算距离，水平的
	C = dist.euclidean(eye[0], eye[3])
	# ear值
	ear = (A + B) / (2.0 * C)
	return ear
	
	
FACIAL_LANDMARKS_68_IDXS = OrderedDict([
	("mouth", (48, 68)),
	("right_eyebrow", (17, 22)),
	("left_eyebrow", (22, 27)),
	("right_eye", (36, 42)),
	("left_eye", (42, 48)),
	("nose", (27, 36)),
	("jaw", (0, 17))
])
(lStart, lEnd) = FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = FACIAL_LANDMARKS_68_IDXS["right_eye"]

#得到68个点的坐标位置
def shape_to_np(shape, dtype="int"):
	# 创建68*2
	coords = np.zeros((shape.num_parts, 2), dtype=dtype)
	# 遍历每一个关键点
	# 得到坐标
	for i in range(0, shape.num_parts):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	return coords

#判断参数，当眼距比大于EYE_AR_THRESH时则为睁眼，否则为闭眼
EYE_AR_THRESH = 0.3

img = cv2.imread("D:/test/mio.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

detector = dlib.get_frontal_face_detector()

#使用预测算子获取得到的人脸区域中的五官的几何点区域，这里加载的是68特征点的landmark模型
predictor = dlib.shape_predictor(
   "C:\\Users\\Coffee\\Anaconda3\\envs\\py36\\Lib\\site-packages\\dlib-data\\shape_predictor_68_face_landmarks.dat"
) 


#找到所有人脸区域
dets = detector(gray, 1)
for face in dets: 
    shape = predictor(gray, face)  # 寻找人脸68个标定点，获取landmark
    shape = shape_to_np(shape)
    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]
    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)
	# 算一个平均的
    ear = (leftEAR + rightEAR) / 2.0
    print(ear)


