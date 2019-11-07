## 人脸识别相关
突发奇想想给陈先生做一组表情包，但是不会P图的我就想到了写代码呀

首先我利用网上一些包写了一些人脸识别的代码，以方便后期我需要获取人脸特征点

## 1. 人脸识别
首先我们要通过人脸识别，获取的是人脸的特征，我们先通过几个方式去获取

### 1.1 利用opencv 进行人脸识别（opencv.py）

 　　Haar特征分类器就是一个XML文件（opencv官方提供了很多分类xml，眼睛鼻子脸部等），存放在OpenCV安装目录中的\data\ haarcascades目录下，
我们这里使用OpenCV提供好的人脸分类模型xml，下载地址：https://github.com/opencv/opencv/tree/master/data/haarcascades 可全部下载到本地，
但是这种方式不是特别的准确


### 1.2 使用dlib进行人脸识别（dlibface.py ）
dlib中使用的是HOG（histogram of oriented gradient）+ 回归树的方法，使用dlib训练好的模型进行检测效果要好很多。dlib也使用了卷积神经网络来进行人脸检测，
效果好于HOG的集成学习方法，不过需要使用GPU加速，不然程序会卡
     
识别阶段。识别也就是我们常说的“分类”，摄像头采集到这个人脸时，让机器判断是张三还是其他人。分类分为两个部分：

    特征向量抽取。本文用到的是dlib中已经训练好的ResNet模型的接口，此接口会返回一个128维的人脸特征向量。
    距离匹配。在获取特征向量之后可以使用欧式距离和本地的人脸特征向量进行匹配，使用最近邻分类器返回样本的标签。

dlib_video.py  视频人脸验证（包括眨眼验证）
	
### 1.3 face_recognition
face_recognition 是使用dlib构建而成，并具有深度学习功能。 
该模型在Labeled Faces in the Wild基准中的准确率为99.38％。并且配备了完整的开发文档和应用案例，特别是兼容树莓派系统。

face_detection.py   利用开源包face_recognition人脸识别

face_recog.py  利用开源包face_recognition人脸验证

face_video.py  视频人脸验证（包括眨眼验证）

#### 1、人脸定位——locations
face_locations(img, number_of_times_to_upsample=1, model="hog")
利用CNN深度学习模型或方向梯度直方图（Histogram of Oriented Gradient, HOG）进行人脸提取。返回值是一个数组(top, right, bottom, left)表示人脸所在边框的四条边的位置。
成员变量：

    img：图片名
    number_of_times_to_upsample=1：数值越大识别的人脸越小，耗时也越长
    model=”hog”：使用的人脸检测模型
    
_raw_face_locations(img, number_of_times_to_upsample=1, model="hog")
_raw_face_locations_batched(images, number_of_times_to_upsample=1, batch_size=128)
batch_face_locations(images, number_of_times_to_upsample=1, batch_size=128)


#### 2、人脸解码——encodings
输入一张图片后，生成一个128维的特征向量，这是 人脸识别的依据。
face_encodings(face_image, known_face_locations=None, num_jitters=1)

    face_image:图片名
    known_face_locations=None:这个是默认值，默认解码图片中的每一个人脸。若输入face_locations()[i]可指定人脸进行解码。
    num_jitters=1：池化操作。数值越高，精度越高，但耗时越长。


#### 3、人脸比对——compare
人脸识别的核心，设置一个阈值，若两张人脸的特征向量的距离，在阈值范围之内，则认为其是同一个人，最后返回一个Boolean值的list。

compare_faces(known_face_encodings, face_encoding_to_check, tolerance=0.6)

    known_face_encodings：人脸解码方法中返回的特征向量，此处是已知人脸的特征向量。
    face_encoding_to_check：此处是未知人脸的特征向量。
    tolerance=0.6：比对阈值，即容差，默认为0.6。

#### 4、人脸特征向量距离——distance
跟上面的方法同样原理，不过输出由对错改为数字。
face_distance(face_encodings, face_to_compare)

    face_encodings：人脸解码方法中返回的特征向量，此处是已知人脸的特征向量。
    face_to_compare：此处是未知人脸的特征向量。
#### 5、人脸特征提取——landmark
face_landmarks(face_image, face_locations=None, model="large")


    face_image：输入图片。
    face_locations=None：这个是默认值，默认解码图片中的每一个人脸。若输入face_locations()[i]可指定人脸进行解码。
    model=”large”：输出的特征模型，默认为“large”，可选“small”，改为输出5点特征，但速度更快。




### 1.4 换脸
人脸识别之后我看了下Procrustes analysis
普氏分析法是一种用来分析形状分布的方法。数学上来讲，就是不断迭代，寻找标准形状(canonical shape)，并利用最小二乘法寻找每个样本形状到这个标准形状的仿射变化方式。（可参照维基百科的GPA算法）
具体可以参考如下链接：
https://blog.csdn.net/tinyzhao/article/details/53169818
https://github.com/mattzheng/Face_Swapping
http://python.jobbole.com/82546/
实现换脸主要是以下几个步骤
1借助 dlib 库检测出图像中的脸部特征
2计算将第二张图像脸部特征对齐到一张图像脸部特征的变换矩阵
3综合考虑两张照片的面部特征获得合适的面部特征掩码
4根据第一张图像人脸的肤色修正第二张图像
5通过面部特征掩码拼接这两张图像
6保存图像

