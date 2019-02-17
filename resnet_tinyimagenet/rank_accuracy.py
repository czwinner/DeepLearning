from config import tiny_imagenet_config as config
from pyimagesearch.preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from pyimagesearch.preprocessing.SimplePreprocessor import SimplePreprocessor
from pyimagesearch.preprocessing.MeanPreprocessor import MeanPreprocessor
from pyimagesearch.utils.ranked import rank5_accuracy
from pyimagesearch.io.hdf5datasetgenerator import HDF5DatasetGenerator
from keras.models import load_model
import json
#加载RGB means for the training set
means=json.loads(open(config.DATASET_MEAN).read())
#初始化image preprocessors
sp=SimplePreprocessor(64,64)
mp=MeanPreprocessor(means["R"],means["G"],means["B"])
iap=ImageToArrayPreprocessor()
#初始化testing dataset generator
testGen=HDF5DatasetGenerator(config.TEST_HDF5,64,
                             preprocessors=[sp,mp,iap],classes=config.NUM_CLASSES)
#加载预训练网络
print("[INFO] loading model...")
model=load_model(config.MODEL_PATH)
#在测试数据上预测
print("[INFO] predicting on test data...")
predictions=model.predict_generator(testGen.generator(),
                                    steps=testGen.numImages // 64,max_queue_size=64*2)
#计算rank-1 和 rank-5准确率
(rank1,rank5)=rank5_accuracy(predictions,testGen.db["labels"])
print("[INFO] rank-1:{:.2f}%".format(rank1*100))
print("[INFO] rank-5:{:.2f}%".format(rank5*100))
#close the database
testGen.close()