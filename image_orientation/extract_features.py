from keras.applications import VGG16
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from sklearn.preprocessing import LabelEncoder
from pyimagesearch.io.hdf5datasetwriter import HDF5DatasetWriter
from imutils import paths
import numpy as np
import progressbar
import argparse
import random
import os
#命令行解析
ap=argparse.ArgumentParser()
ap.add_argument("-d","--dataset",required=True,
                help="path to input dataset")
ap.add_argument("-o","--output",required=True,
                help="path to output HDF5 file")
ap.add_argument("-b","--batch-size",type=int,default=32,
                help="batch size of images to be passed through network")
ap.add_argument("-s","--buffer-size",type=int,default=1000,
                help="size of feature extraction buffer")
args=vars(ap.parse_args())
#存储batch size
bs=args["batch_size"]
#抓取图像列表将它们混洗以允许在训练期间通过阵列切片进行简单的训练和测试分割
print("[INFO] loading images...")
imagePaths=list(paths.list_images(args["dataset"]))
random.shuffle(imagePaths)
#提取类标签然后encode
labels=[p.split(os.path.sep)[-2] for p in imagePaths]
le=LabelEncoder()
labels=le.fit_transform(labels)
#加载VGG16
model=VGG16(weights="imagenet",include_top=False)
#初始化HDF5 dataset writer,然后保存类标签名字
dataset=HDF5DatasetWriter((len(imagePaths),512*7*7),
                          args["output"],dataKey="features",bufSize=args["buffer_size"])
dataset.storeClassLabels(le.classes_)
#初始化progress bar
widgets=["Extracting Features: ",progressbar.Percentage()," ",
         progressbar.Bar()," ",progressbar.ETA()]
pbar=progressbar.ProgressBar(maxval=len(imagePaths),
                             widgets=widgets).start()
#loop over the images in patches
for i in np.arange(0,len(imagePaths),bs):
    #提取一批图像和列表和标签，然后初始化将通过网络传递以进行特征提取的实际图像列表
    batchPaths=imagePaths[i:i+bs]
    batchLabels=labels[i:i+bs]
    batchImages=[]
    #循环当前批次的图像和标签
    for (j,imagePath) in enumerate(batchPaths):
        #加载图像用keras实用程序把图像变为224x224
        image=load_img(imagePath,target_size=(224,224))
        image=img_to_array(image)
        #处理图像 (1)增加维度 (2)减去ImageNet dataset的RGB均值
        image=np.expand_dims(image,axis=0)
        image=imagenet_utils.preprocess_input(image)
        # add the image to the batch
        batchImages.append(image)
    #通过网络传递图像并使用输出作为实际特征
    batchImages=np.vstack(batchImages)
    features=model.predict(batchImages,batch_size=bs)
    #重塑特征，使每个图像由'MaxPooling2D' 输出的平面特征向量表示
    features=features.reshape((features.shape[0],512*7*7))
    #把特征和标签加入HDF5 dataset
    dataset.add(features,batchLabels)
    pbar.update(i)
#关闭数据集
dataset.close()
pbar.finish()
