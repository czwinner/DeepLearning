from keras.applications import VGG16
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from imutils import paths
import numpy as np
import argparse
import pickle
import imutils
import h5py
import cv2
#命令行解析
ap=argparse.ArgumentParser()
ap.add_argument("-d","--db",required=True,help="path HDF5 data base")
ap.add_argument("-i","--dataset",required=True,
                help="path to the input images dataset")
ap.add_argument("-m","--model",required=True,
                help="path to trained orientation model")
args=vars(ap.parse_args())
#从HDF5 数据库加载标签名
db=h5py.File(args["db"])
labelNames=[int(angle) for angle in db["label_names"][:]]
db.close()
#抓取测试图像的路径然后随机取10张图片
print("[INFO] sampling images...")
imagePaths=list(paths.list_images(args["dataset"]))
imagePaths=np.random.choice(imagePaths,size=(10,),replace=False)
#加载VGG16
print("[INFO] loading network...")
vgg=VGG16(weights="imagenet",include_top=False)
#加载orientation model
print("[INFO] loading model...")
model=pickle.loads(open(args["model"],"rb").read())
#循环图像路径
for imagePath in imagePaths:
    #加载图像
    orig=cv2.imread(imagePath)
    #用keras实用程序使图像变为224x224
    image = load_img(imagePath, target_size=(224, 224))
    image=img_to_array(image)
    #处理图像(1)增加维度 (2)减去图像RGB像素平均值
    image=np.expand_dims(image,axis=0)
    image=imagenet_utils.preprocess_input(image)
    #使图像通过网络得到特征向量
    features=vgg.predict(image)
    features=features.reshape((features.shape[0],512*7*7))
    #通过分类得到方向的预测值
    angle=model.predict(features)
    angle=labelNames[angle[0]]
    #现在已经预测了图像的方向，可以对其进行校正
    rotated=imutils.rotate_bound(orig,360-angle)
    #显示原始和纠正了的图像
    cv2.imshow("Original",orig)
    cv2.imshow("Corrected",rotated)
    cv2.waitKey(0)