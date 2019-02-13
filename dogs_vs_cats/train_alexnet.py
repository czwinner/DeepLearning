import matplotlib
matplotlib.use("Agg")
from config import dogs_vs_cats_config as config
from pyimagesearch.preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from pyimagesearch.preprocessing.SimplePreprocessor import SimplePreprocessor
from pyimagesearch.preprocessing.patchpreprocessor import PatchPreprocessor
from pyimagesearch.preprocessing.MeanPreprocessor import MeanPreprocessor
from pyimagesearch.callbacks.trainingmonitor import TrainingMonitor
from pyimagesearch.io.hdf5datasetgenerator import HDF5DatasetGenerator
from pyimagesearch.nn.conv.alexnet import AlexNet
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
import json
import os
#构造图像增强
aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
                         width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
                         horizontal_flip=True, fill_mode="nearest")
#加载RGB的平均值
means=json.loads(open(config.DATASET_MEAN).read())
#初始化图像预处理器
sp=SimplePreprocessor(227, 227)
pp=PatchPreprocessor(227, 227)
mp=MeanPreprocessor(means["R"], means["G"], means["B"])
iap=ImageToArrayPreprocessor()
#初始化训练集和验证集的数据集生成器
trainGen=HDF5DatasetGenerator(config.TRAIN_HDF5, 128, aug=aug,
                              preprocessors=[pp, mp, iap], classes=2)
valGen=HDF5DatasetGenerator(config.VAL_HDF5, 128,
                            preprocessors=[sp, mp, iap], classes=2)
#初始化优化器
print("[INFO] compiling model...")
opt=Adam(lr=1e-3)
model=AlexNet.build(width=227, height=227, depth=3, classes=2, reg=0.0002)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
#构建一组回调函数
path=os.path.sep.join([config.OUTPUT_PATH, "{}.png".format(os.getpid())])
callbacks = [TrainingMonitor(path)]
#训练网络
model.fit_generator(
    trainGen.generator(),
    steps_per_epoch=trainGen.numImages // 128,
    validation_data=valGen.generator(),
    validation_steps=valGen.numImages // 128,
    epochs=75,
    max_queue_size=128*2,
    callbacks=callbacks, verbose=1
)
#保存模型
print("[INFO] serializing model...")
model.save(config.MODEL_PATH, overwrite=True)
#关闭HDF5数据库
trainGen.close()
valGen.close()