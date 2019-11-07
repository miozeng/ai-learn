# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 17:21:00 2019

@author: Coffee
"""

# -*- coding: utf-8 -*-
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

def shape_to_np(shape, dtype="int"):
	# 创建68*2
	coords = np.zeros((shape.num_parts, 2), dtype=dtype)
	# 遍历每一个关键点
	# 得到坐标
	for i in range(0, shape.num_parts):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	return coords

threshold = 0.54
def findNearestClassForImage(face_descriptor, faceLabel):
    #脸部跟之前的每一个脸部二维码比对
    temp = face_descriptor - data
    e = np.linalg.norm(temp,axis=1,keepdims=True)
    #得到距离最小的数据
    min_distance = e.min()
    print('distance: ', min_distance)
    #如果距离大于最大的阈值则找不到对象
    if min_distance > threshold:
        return 'other'
    index = np.argmin(e)
    return faceLabel[index]

#===========================获取用户编码，这一步应该从数据库去读的，暂时省去===============
data = np.zeros((1,128))                                                                            #定义一个128维的空向量data
label = []                                                                                          #定义空的list存放人脸的标签


img = cv2.imread("D:/test/mio.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#人脸分类器
detector = dlib.get_frontal_face_detector()

#使用预测算子获取得到的人脸区域中的五官的几何点区域，这里加载的是68特征点的landmark模型
predictor = dlib.shape_predictor(
   "C:\\Users\\Coffee\\Anaconda3\\envs\\py36\\Lib\\site-packages\\dlib-data\\shape_predictor_68_face_landmarks.dat"
) 

#找到所有人脸区域
dets = detector(gray, 1)

#得到ResNet模型
facerec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
#找到所有人脸区域
dets = detector(gray, 1)
for face in dets:
    shape = predictor(img, face)  # 寻找人脸68个标定点，获取landmark
   
    #如果有需要可以获得128维的人脸特征向量保存
    face_descriptor = facerec.compute_face_descriptor(img, shape)  # 使用resNet获取128维的人脸特征向量
    faceArray = np.array(face_descriptor).reshape((1, 128))  # 转换成numpy中的数据结构
    data = np.concatenate((data, faceArray))
    label.append("mio")
    #保存代码省略

#=======================初始化参数=============================
#人物是否已识别
reUser = False;
#是否已眨眼
userEyeClose = False;
#是否已睁眼
userEyeOpen = False;

#判断参数，当眼距比大于EYE_AR_THRESH时则为睁眼，否则为闭眼
EYE_AR_THRESH = 0.3
	
#===============开始处理视频=====================
video_capture = cv2.VideoCapture(0)


while True:
    #1.读取视频，获取图片，将图片缩小
    ret, frame = video_capture.read()

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

   
    
    #获取视频人脸   
    dets = detector(small_frame, 1)
    for face in dets:
        shape = predictor(small_frame, face)
        #将脸部框起来
        cv2.rectangle(small_frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
        
         #2.判断是否为当前用户
        if not reUser:
            face_descriptor = facerec.compute_face_descriptor(small_frame, shape)
            faceLabel = findNearestClassForImage(face_descriptor,label)
            print(faceLabel) 
            reUser = True
        
        #3.查找图像中所有面部的所有面部特征,判断眼距
        shape = shape_to_np(shape)
        # 分别计算ear值
    	leftEye = shape[lStart:lEnd]
    	rightEye = shape[rStart:rEnd]
    	leftEAR = eye_aspect_ratio(leftEye)
    	rightEAR = eye_aspect_ratio(rightEye)
    	# 算一个平均的
    	ear = (leftEAR + rightEAR) / 2.0
        print(ear)
        if (ear>=EYE_AR_THRESH):
            userEyeOpen = True
        else:
            userEyeClose = True
    

	
    cv2.imshow('Video', frame)
    #如果人物已经验证成功则退出
    if reUser & userEyeClose & userEyeOpen:
        print("validate ")
        break
		
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
