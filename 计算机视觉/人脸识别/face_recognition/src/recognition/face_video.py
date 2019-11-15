# -*- coding: utf-8 -*-
import face_recognition
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
import argparse
import time
import dlib
import cv2


#1.读取视频，获取图片，将图片缩小
#2.获取视频人脸，判断是否为当前用户
#3.获取视频图片，直到判断到眼睛小于某个值，证明有闭眼，大于某个值证明有睁眼



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


#===========================获取用户编码，这一步应该从数据库去读的，暂时省去===============
image = face_recognition.load_image_file("D:/test/mio.jpg")
mio_face_encoding = face_recognition.face_encodings(image)[0]

#=======================初始化参数=============================
#人物是否已识别
reUser = False;
#是否已眨眼
userEyeClose = False;
#是否已睁眼
userEyeOpen = False;

#判断参数，当眼距比大于EYE_AR_THRESH时则为睁眼，否则为闭眼
EYE_AR_THRESH = 0.3
	
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

	
video_capture = cv2.VideoCapture(0)


while True:
    #1.读取视频，获取图片，将图片缩小
    ret, frame = video_capture.read()

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

	#验证人物，以及验证眼距
    #if process_this_frame:
    #2.获取视频人脸，判断是否为当前用户
    if not reUser:
        face_locations = face_recognition.face_locations(small_frame)
        face_encodings = face_recognition.face_encodings(small_frame, face_locations)
    
        face_names = []
        for face_encoding in face_encodings:
            match = face_recognition.compare_faces([mio_face_encoding], face_encoding)
    
            if match[0]:
                name = "mio"
                print('hello mio~')
                reUser = True
            else:
                name = "unknown"
    
            face_names.append(name)

	#3.查找图像中所有面部的所有面部特征,判断眼距
    face_landmarks_list = face_recognition.face_landmarks(small_frame)
    for face_landmarks in face_landmarks_list:
        leftEye = face_landmarks['left_eye']
        rightEye = face_landmarks['right_eye']
        leftEAR = eye_aspect_ratio(leftEye)
        
        rightEAR = eye_aspect_ratio(rightEye)           
        ear = (leftEAR + rightEAR) / 2.0
        print(ear)
        if (ear>=EYE_AR_THRESH):
            userEyeOpen = True
        else:
            userEyeClose = True
                
          
		
    #process_this_frame = not process_this_frame
	
	#将人物画出来
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

	
    cv2.imshow('Video', frame)
    #如果人物已经验证成功则退出
    if reUser & userEyeClose & userEyeOpen:
        print("validate ")
        break
		
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
