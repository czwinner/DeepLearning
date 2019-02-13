from config import dogs_vs_cats_config as config
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pyimagesearch.preprocessing.aspectawarepreprocessor import AspectAwarePreprocessor
from pyimagesearch.io.hdf5datasetwriter import HDF5DatasetWriter
from imutils import paths
import numpy as np
import progressbar
import json
import cv2
import os
#抓取图像路径
trainPaths=list(paths.list_images(config.IMAGES_PATH))
trainLabels=[p.split(os.path.sep)[1].split(".")[0] for p in trainPaths]
le=LabelEncoder()
trainLabels=le.fit_transform(trainLabels)
#从训练集中执行分层抽样，以根据训练数据简历测试集
split=train_test_split(trainPaths,trainLabels,test_size=config.NUM_TEST_IMAGES,
                       stratify=trainLabels,random_state=42)
(trainPaths,testPaths,trainLabels,testLabels)=split
#执行另一个分层抽样，建立验证集
split=train_test_split(trainPaths,trainLabels,test_size=config.NUM_VAL_IMAGES,
                       stratify=trainLabels,random_state=42)
(trainPaths,valPaths,trainLabels,valLabels)=split
#构建一个列表，配对训练，验证和测试图像路径及相应的标签和输出HDF5文件
datasets=[
    ("train",trainPaths,trainLabels,config.TRAIN_HDF5),
    ("val",valPaths,valLabels,config.VAL_HDF5),
    ("test",testPaths,testLabels,config.TEST_HDF5),
]
#初始化图像预处理和RGB平均值列表
aap=AspectAwarePreprocessor(256,256)
(R,G,B)=([],[],[])
#循环dataset元组
for (dType,paths,labels,outputPath) in datasets:
    #创建HDF5 writer
    print("[INFO] building {}...".format(outputPath))
    writer=HDF5DatasetWriter((len(paths),256,256,3),outputPath)
    #初始化progress bar
    widgets=["Building Dataset:",progressbar.Percentage()," ",
             progressbar.Bar()," ",progressbar.ETA()]
    pbar=progressbar.ProgressBar(maxval=len(paths),
                                 widgets=widgets).start()
    #循环image,paths
    for (i,(path,label)) in enumerate(zip(paths,labels)):
        #加载图片处理
        image=cv2.imread(path)
        image=aap.preprocess(image)
        #如果正在构建训练集，则计算图像中每个通道的平均值，然后更新相应列表
        if dType=="train":
            (b,g,r)=cv2.mean(image)[:3]
            R.append(r)
            G.append(g)
            B.append(b)
        #把图像和标签写入HDF5 dataset
        writer.add([image],[label])
        pbar.update(i)
    #关闭HDF5 writer
    pbar.finish()
    writer.close()
#构造一个颜色平均值字典，然后将平均值序列化为JSON文件
print("[INFO] serializing means...")
D={"R":np.mean(R),"G":np.mean(G),"B":np.mean(B)}
f=open(config.DATASET_MEAN,"w")
f.write(json.dumps(D))
f.close()
