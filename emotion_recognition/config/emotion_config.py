from os import path
#定义数据集的基本路径
BASE_PATH="datasets/fer2013"
#定义csv路径
INPUT_PATH=path.sep.join([BASE_PATH,"fer2013/fer2013.csv"])
#定义类别数
NUM_CLASSES=6
#定义HDF5文件路径
TRAIN_HDF5=path.sep.join([BASE_PATH,"hdf5/train.hdf5"])
VAL_HDF5=path.sep.join([BASE_PATH,"hdf5/val.hdf5"])
TEST_HDF5=path.sep.join([BASE_PATH,"hdf5/test.hdf5"])
#定义batch size
BATCH_SIZE=128
#定义output 路径
OUTPUT_PATH=path.sep.join([BASE_PATH,"output"])
