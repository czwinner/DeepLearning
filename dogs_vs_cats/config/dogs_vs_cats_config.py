#定义图像路径
IMAGES_PATH="train"
#由于我们没有验证集数据或访问测试标签，因此需要从训练数据中获取图像并使用它们
NUM_CLASSES=2
NUM_VAL_IMAGES=1250*NUM_CLASSES
NUM_TEST_IMAGES=1250*NUM_CLASSES
#定义training,validation和testing的HDF5文件路径
TRAIN_HDF5="hdf5/train.hdf5"
VAL_HDF5="hdf5/val.hdf5"
TEST_HDF5="hdf5/test.hdf5"
#定义输出的model文件路径
MODEL_PATH="output/alexnet_dogs_vs_cats.model"
#定义数据颜色均值的路径
DATASET_MEAN="output/dogs_vs_cats_mean.json"
#定义用于存储绘图，分类报告等的输出目录路径
OUTPUT_PATH="output"
