# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 10:30:21 2019

@author: Coffee
"""
import numpy as np
import cv2

def cv_show(name,img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detectAndDescribe( image):
    # 将彩色图片转换成灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create()
    print('start to get detect')
    # find the keypoints with ORB
    kps = orb.detect(image,None)
    # compute the descriptors with ORB
    (kps, features) = orb.compute(image, kps)
    # 检测SIFT特征点，并计算描述子
   # (kps, features) = orb.detectAndCompute(image, None)
    # 将结果转换成NumPy数组
    kps = np.float32([kp.pt for kp in kps])

    # 返回特征点集，及对应的描述特征
    return (kps, features)

def matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh):
    # 建立暴力匹配器
    matcher = cv2.BFMatcher()
  
    # 使用KNN检测来自A、B图的SIFT特征匹配对，K=2
    rawMatches = matcher.knnMatch(featuresA, featuresB, 2)

    matches = []
    for m in rawMatches:
        # 当最近距离跟次近距离的比值小于ratio值时，保留此匹配对
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
        # 存储两个点在featuresA, featuresB中的索引值
            matches.append((m[0].trainIdx, m[0].queryIdx))

    # 当筛选后的匹配对大于4时，计算视角变换矩阵
    if len(matches) > 4:
        # 获取匹配对的点坐标
        ptsA = np.float32([kpsA[i] for (_, i) in matches])
        ptsB = np.float32([kpsB[i] for (i, _) in matches])

        # 计算视角变换矩阵
        (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)

        # 返回结果
        return (matches, H, status)

    # 如果匹配对小于4时，返回None
    return None
#图片拼接
#1.读入图片
imageA = cv2.imread("img/left_01.png")
imageB = cv2.imread("img/right_01.png")

ratio=0.75 
reprojThresh=4.0
#2.获取图片特征点
(kpsA, featuresA) = detectAndDescribe(imageA)
(kpsB, featuresB) = detectAndDescribe(imageB)
#3.暴力匹配图片
M = matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh)

# 如果返回结果为空，没有匹配成功的特征点，退出算法
#if M is None:
#    return 

# 否则，提取匹配结果
# H是3x3视角变换矩阵      
(matches, H, status) = M

#4.将图片A做视角转换
result = cv2.warpPerspective(imageA, H, (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
cv_show('result', result)
      
#5.图片拼接
# 将图片B传入result图片最左端
result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB
cv_show('result2', result)
