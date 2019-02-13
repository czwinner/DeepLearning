import matplotlib
matplotlib.use("Agg")
from config import emotion_config as config
from pyimagesearch.preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from pyimagesearch.callbacks.trainingmonitor import TrainingMonitor
from pyimagesearch.io.hdf5datasetgenerator import HDF5DatasetGenerator
from pyimagesearch.nn.conv.emotionvggnet import EmotionVGGNet
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.models import load_model
import keras.backend as K
import argparse
import os
#命令行解析
ap=argparse.ArgumentParser()
ap.add_argument("-m","--model",type=str,
                help="path to *specific* model checkpoint to load")
args=vars(ap.parse_args())
#训练和测试集的图像增强，初始化图像预处理
trainAug=ImageDataGenerator(rotation_range=10,zoom_range=0.1,
                            horizontal_flip=True,rescale=1/255.0,fill_mode="nearest")
valAug=ImageDataGenerator(rescale=1/255.0)
iap=ImageToArrayPreprocessor()
#初始化训练和验证的dataset generator
trainGen=HDF5DatasetGenerator(config.TRAIN_HDF5,config.BATCH_SIZE,
                              aug=trainAug,preprocessors=[iap],classes=config.NUM_CLASSES)
valGen=HDF5DatasetGenerator(config.VAL_HDF5,config.BATCH_SIZE,
                            aug=valAug,preprocessors=[iap],classes=config.NUM_CLASSES)
print("[INFO] compiling model...")
model=EmotionVGGNet.build(width=48,height=48,depth=1,
                              classes=config.NUM_CLASSES)
opt=Adam(lr=1e-3)
model.compile(loss="categorical_crossentropy",optimizer=opt,
                  metrics=["accuracy"])
#构建回调函数
figPath=os.path.sep.join([config.OUTPUT_PATH,"vggnet_emotion.png"])
jsonPath=os.path.sep.join([config.OUTPUT_PATH,"vggnet_emotion.json"])
callbacks=[TrainingMonitor(figPath,jsonPath=jsonPath)]
#训练
model.fit_generator(
    trainGen.generator(),
    steps_per_epoch=trainGen.numImages // config.BATCH_SIZE,
    validation_data=valGen.generator(),
    validation_steps=valGen.numImages // config.BATCH_SIZE,
    epochs=75,
    max_queue_size=config.BATCH_SIZE*2,
    callbacks=callbacks,verbose=1
)
#关闭database
trainGen.close()
valGen.close()
model.save(args["model"])

