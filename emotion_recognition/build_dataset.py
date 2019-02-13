from config import emotion_config as config
from pyimagesearch.io.hdf5datasetwriter import HDF5DatasetWriter
import numpy as np
#打开csv文件（跳过标题行),初始化训练、验证、测试集的数据列表和标签列表
print("[INFO] loading input data...")
f=open(config.INPUT_PATH)
f.__next__()
(trainImages,trainLabels)=([],[])
(valImages,valLabels)=([],[])
(testImages,testLabels)=([],[])
#循环csv文件
for row in f:
    #提取标签，图像和用法
    (label,image,usage)=row.strip().split(",")
    label=int(label)
    #如果忽略 "disgust"则总共为6个标签
    if config.NUM_CLASSES==6:
        #合并"anger"和"disgust"标签
        if label==1:
            label=0
        #如果label大于0，则减去1，使得所有标签顺序
        if label > 0:
            label-=1
    # reshape the flatten pixel into a 48x48 image
    image=np.array(image.split(" "),dtype="uint8")
    image=image.reshape((48,48))
    #查看是否是训练图像
    if usage == "Training":
        trainImages.append(image)
        trainLabels.append(label)
    #查看是否是验证图像
    elif usage == "PrivateTest":
        valImages.append(image)
        valLabels.append(label)
    #否则是测试图像
    else:
        testImages.append(image)
        testLabels.append(label)
#构建训练，验证，测试图像和他们标签和hdf5的元组列表
datasets=[
    (trainImages,trainLabels,config.TRAIN_HDF5),
    (valImages,valLabels,config.VAL_HDF5),
    (testImages,testLabels,config.TEST_HDF5)]
#循环dataset元组
for (images,labels,outputPath) in datasets:
    #创建HDF5 writer
    print("[INFO] building {}...".format(outputPath))
    writer=HDF5DatasetWriter((len(images),48,48),outputPath)
    #循环图像，把它们加入dataset
    for (image,label) in zip(images,labels):
        writer.add([image],[label])
    #关闭HDF5 writer
    writer.close()
#关闭 csv 文件
f.close()
