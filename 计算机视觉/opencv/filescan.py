# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 10:27:47 2019

@author: Coffee
"""
#文件扫描与OCR
#文件扫描的前提是我们将图片变换成插件pytesseract可读的图片
import cv2
import numpy as np


def cv_show(name,img):
    cv2.imshow(name,img)
    cv2.waitKey()
    cv2.destroyAllWindows()
def order_points(pts):
	# 一共4个坐标点
	rect = np.zeros((4, 2), dtype = "float32")

	# 按顺序找到对应坐标0123分别是 左上，右上，右下，左下
	# 计算左上，右下
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]

	# 计算右上和左下
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]

	return rect

def four_point_transform(image, pts):
	# 获取输入坐标点
	rect = order_points(pts)
	(tl, tr, br, bl) = rect

	# 计算输入的w和h值
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))

	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))

	# 变换后对应坐标位置
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")

	# 计算变换矩阵
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

	# 返回变换后结果
	return warped   
def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
	dim = None
	(h, w) = image.shape[:2]
	if width is None and height is None:
		return image
	if width is None:
		r = height / float(h)
		dim = (int(w * r), height)
	else:
		r = width / float(w)
		dim = (width, int(h * r))
	resized = cv2.resize(image, dim, interpolation=inter)
	return resized
#1.读入图片
image = cv2.imread("img/page.jpg")

#2.将图片改成高度为500的图片
orig = image.copy()
(h,w) = orig.shape[:2]
height = 500
r = height / float(h)
image = cv2.resize(orig,  (int(w * r), height), interpolation=cv2.INTER_AREA)

print(orig.shape[:2])
print(image.shape[:2])
#3.预处理图片
#3.1获取灰度图
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#3.2图片平滑以减少噪音
gray = cv2.GaussianBlur(gray, (5, 5), 0)
#3.3获取图片轮廓
edged = cv2.Canny(gray, 75, 200)

#3.4展示结果
cv_show("image", image)
cv_show("edged", edged)

#4.轮廓检测
cnts, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]
#5.遍历较大的轮廓获取有四个点的轮廓（则为我们的发票本身）
for c in cnts:

	# 计算轮廓近似
	peri = cv2.arcLength(c, True)
	# C表示输入的点集
	# epsilon表示从原始轮廓到近似轮廓的最大距离，它是一个准确度参数
	# True表示封闭的
	approx = cv2.approxPolyDP(c, 0.02 * peri, True)

	# 4个点的时候就拿出来
	if len(approx) == 4:
		screenCnt = approx
		break

#5.2.展示结果
print('show Contours')
cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
cv_show("Outline", image)

#6.透视变化得到对应的小票的图片
ratio = orig.shape[0] / 500.0
warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
#7.二值处理得到较为清晰的小票内容
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
ref = cv2.threshold(warped, 100, 255, cv2.THRESH_BINARY)[1]
cv2.imwrite('img/scan.jpg', ref)

cv_show("Original", resize(orig, height = 650))
cv_show("Scanned", resize(ref, height = 650))
