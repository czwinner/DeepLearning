#设置maplotlib后端
import matplotlib
matplotlib.use("Agg")
from config import tiny_imagenet_config as config
from pyimagesearch.preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from pyimagesearch.preprocessing.SimplePreprocessor import SimplePreprocessor
from pyimagesearch.preprocessing.MeanPreprocessor import MeanPreprocessor
from pyimagesearch.callbacks.trainingmonitor import TrainingMonitor
from pyimagesearch.io.hdf5datasetgenerator import HDF5DatasetGenerator
from pyimagesearch.nn.conv.resnet import ResNet
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD
import argparse
import json
import os
import sys
#定义epoch总数和初始学习率
NUM_EPOCHS=75
INIT_LR=1e-1
def poly_decay(epoch):
    #初始化最大epoch,基础学习率和多项式幂
    maxEpochs=NUM_EPOCHS
    baseLR=INIT_LR
    power=1.0
    #计算新的学习率
    alpha=baseLR*(1-(epoch / float(maxEpochs)))**power
    #返回alpha
    return alpha
#命令行解析
ap=argparse.ArgumentParser()
ap.add_argument("-m","--model",required=True,
                help="path to output model")
ap.add_argument("-o","--output",required=True,
                help="path to output directory (logs,plots,etc)")
args=vars(ap.parse_args())
#图像增强
aug=ImageDataGenerator(rotation_range=18,zoom_range=0.15,
                       width_shift_range=0.2,height_shift_range=0.2,
                       shear_range=0.15,horizontal_flip=True,fill_mode="nearest")
#加载RGB平均值
means=json.loads(open(config.DATASET_MEAN).read())
#初始化图形预处理器
sp=SimplePreprocessor(64,64)
mp=MeanPreprocessor(means["R"],means["G"],means["B"])
iap=ImageToArrayPreprocessor()
#初始化traning and validation dataset generator
trainGen=HDF5DatasetGenerator(config.TRAIN_HDF5,64,aug=aug,
                              preprocessors=[sp,mp,iap],classes=config.NUM_CLASSES)
valGen=HDF5DatasetGenerator(config.VAL_HDF5,64,
                            preprocessors=[sp,mp,iap],classes=config.NUM_CLASSES)
#构建回调函数
figPath=os.path.sep.join([args["output"],"{}.png".format(
    os.getpid())])
jsonPath=os.path.sep.join([args["output"],"{}.png".format(
    os.getpid())])
callbacks=[TrainingMonitor(figPath,jsonPath=jsonPath),
           LearningRateScheduler(poly_decay)]
#初始化优化器和模型
print("[INFO] compiling model...")
model=ResNet.build(64,64,3,config.NUM_CLASSES,(3,4,6),
                   (64,128,256,512),reg=0.0005,dataset="tiny_imagenet")
opt=SGD(lr=INIT_LR,momentum=0.9)
model.compile(loss="categorical_crossentropy",optimizer=opt,
              metrics=["accuracy"])
#训练
print("[INFO] trainging network...")
model.fit_generator(
    trainGen.generator(),
    steps_per_epoch=trainGen.numImages // 64,
    validation_data=valGen.generator(),
    validation_steps=valGen.numImages // 64,
    epochs=NUM_EPOCHS,
    max_queue_size=64*2,
    callbacks=callbacks,verbose=1
)
#保存模型
print("[INFO] serializing network...")
model.save(args["model"])
#关闭databases
trainGen.close()
valGen.close()