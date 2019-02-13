from config import tiny_imagenet_config as config
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pyimagesearch.io.hdf5datasetwriter import HDF5DatasetWriter
from imutils import paths
import numpy as np
import progressbar
import json
import cv2
import os
#提取训练图像路径，提取训练类标签并对其进行编码
trainPaths=list(paths.list_images(config.TRAIN_IMAGES))
trainLabels=[p.split(os.path.sep)[-3] for p in trainPaths]
le=LabelEncoder()
trainLabels=le.fit_transform(trainLabels)
#从训练集中执行分层抽样以构建测试集
split=train_test_split(trainPaths,trainLabels,test_size=config.NUM_TEST_IMAGES,
                       stratify=trainLabels,random_state=42)
(trainPaths,testPaths,trainLabels,testLabels)=split
#从文件加载validation filename=> class,然后用这些映射构建验证路径和标签列表
M=open(config.VAL_MAPPINGS).read().strip().split("\n")
M=[r.split("\t")[:2] for r in M]
valPaths=[os.path.sep.join([config.VAL_IMAGES,m[0]]) for m in M]
valLabels=le.transform([m[1] for m in M])
#构建一个training,validation and test的图像，标签和输出HDF5路径的元组列表
datasets=[
    ("train",trainPaths,trainLabels,config.TRAIN_HDF5),
    ("val",valPaths,valLabels,config.VAL_HDF5),
    ("test",testPaths,testLabels,config.TEST_HDF5)
]
#初始化RGB通道平均值的列表
(R,G,B)=([],[],[])
#循环datasets元组
for (dType,paths,labels,outputPath) in datasets:
    #创建HDF5 writer
    print("[INFO] building {}...".format(outputPath))
    writer=HDF5DatasetWriter((len(paths),64,64,3),outputPath)
    #初始化progress bar
    widgets=["Building Dataset: ",progressbar.Percentage()," ",
             progressbar.Bar()," ",progressbar.ETA()]
    pbar=progressbar.ProgressBar(maxval=len(paths),widgets=widgets).start()
    #循环图像路径
    for (i,(path,label)) in enumerate(zip(paths,labels)):
        #加载图像
        image=cv2.imread(path)
        #如果正在构建训练数据集，则计算图像中每个通道平均值，然后更新相应列表
        if dType=="train":
            (b,g,r)=cv2.mean(image)[:3]
            R.append(r)
            G.append(g)
            B.append(b)
        #将图像和标签加入HDF5 dataset
        writer.add([image],[label])
        pbar.update(i)
    #关闭HDF5 writer
    pbar.finish()
    writer.close()
#构造一个平均字典，然后将平局值序列化为JSON文件
print("[INFO] serializing means...")
D={"R":np.mean(R),"G":np.mean(G),"B":np.mean(B)}
f=open(config.DATASET_MEAN,"w")
f.write(json.dumps(D))
f.close()