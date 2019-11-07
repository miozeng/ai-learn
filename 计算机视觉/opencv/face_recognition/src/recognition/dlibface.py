import cv2
import dlib
import numpy as np
import os
import json

data = np.zeros((1,128))                                                                            #定义一个128维的空向量data
label = []                                                                                          #定义空的list存放人脸的标签


filepath = "D:\\test2\\wz.jpg"
img = cv2.imread(filepath)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#人脸分类器
#传统的HOG特征+级联分类的方法
detector = dlib.get_frontal_face_detector()
#基于CNN的方式
#detector = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')

#使用预测算子获取得到的人脸区域中的五官的几何点区域，这里加载的是68特征点的landmark模型
predictor = dlib.shape_predictor(
   "D:\\Program Files\\python3.6\\Lib\\site-packages\\dlib-data\\shape_predictor_68_face_landmarks.dat"
)
#得到ResNet模型
facerec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

#找到所有人脸区域
dets = detector(gray, 1)
for face in dets:
    shape = predictor(img, face)  # 寻找人脸68个标定点，获取landmark
    #将脸部框起来
    #cv2.rectangle(img, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
    # 遍历所有点，打印出其坐标，并圈出来
    for pt in shape.parts():
        pt_pos = (pt.x, pt.y)
        cv2.circle(img, pt_pos, 2, (0, 255, 0), 1)
    cv2.imshow("image", img)
    #如果有需要可以获得128维的人脸特征向量保存
    face_descriptor = facerec.compute_face_descriptor(img, shape)  # 使用resNet获取128维的人脸特征向量
    faceArray = np.array(face_descriptor).reshape((1, 128))  # 转换成numpy中的数据结构
    data = np.concatenate((data, faceArray))
    label.append("this user's lable")
    #保存代码省略

cv2.waitKey(0)
cv2.destroyAllWindows()


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
