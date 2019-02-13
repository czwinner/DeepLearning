from config import dogs_vs_cats_config as config
from pyimagesearch.preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from pyimagesearch.preprocessing.SimplePreprocessor import SimplePreprocessor
from pyimagesearch.preprocessing.MeanPreprocessor import MeanPreprocessor
from pyimagesearch.preprocessing.croppreprocessor import CropPreprocessor
from pyimagesearch.io.hdf5datasetgenerator import HDF5DatasetGenerator
from pyimagesearch.utils.ranked import rank5_accuracy
from keras.models import load_model
import numpy as np
import progressbar
import json
#加载RGB平均值为训练集
means=json.loads(open(config.DATASET_MEAN).read())
#初始化图像预处理器
sp=SimplePreprocessor(227,227)
mp=MeanPreprocessor(means["R"],means["G"],means["B"])
cp=CropPreprocessor(227,227)
iap=ImageToArrayPreprocessor()
#加载预训练模型
print("[INFO] loading model...")
model=load_model(config.MODEL_PATH)
#初始化testing dataset generator,然后预测测试集
print("[INFO] predicting on test data (no crops) ...")
testGen=HDF5DatasetGenerator(config.TEST_HDF5,64,
                             preprocessors=[sp,mp,iap],classes=2)
predictions=model.predict_generator(testGen.generator(),
                                    steps=testGen.numImages // 64,max_queue_size=64*2)
#计算 rank-1和rank-5准确率
(rank1,_)=rank5_accuracy(predictions,testGen.db["labels"])
print("[INFO] rank-1: {:.2f} %".format(rank1*100))
testGen.close()
#重新初始化testing generator 这次不包括'Simple Preprocessor'
testGen=HDF5DatasetGenerator(config.TEST_HDF5,64,
                             preprocessors=[mp],classes=2)
predictions=[]
#初始化progressbar
widgets=["Evaluating: ",progressbar.Percentage()," ",
         progressbar.Bar()," ",progressbar.ETA()]
pbar=progressbar.ProgressBar(maxval=testGen.numImages // 64,
                             widgets=widgets).start()
#循环测试数据的一个pass
for (i,(images,labels)) in enumerate(testGen.generator(passes=1)):
    #循环每一个图像
    for images in images:
        #将剪裁预处理器应用于图像以生成10个单独的剪裁，然后将它们以图像转换为数组
        crops=cp.preprocess(images)
        crops=np.array([iap.preprocess(c) for c in crops],dtype="float32")
        #对剪裁的图像预测，然后将它们平均在一起以获得最终预测
        pred=model.predict(crops)
        predictions.append(pred.mean(axis=0))
    #更新progress bar
    pbar.update(i)
#计算rank-1准确率
pbar.finish()
print("[INFO] predictiong on test data (with crops)...")
(rank1,_)=rank5_accuracy(predictions,testGen.db["labels"])
print("[INFO] rank-1:{:.2f} %".format(rank1*100))
testGen.close()
