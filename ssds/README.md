# 概述
该项目使用Tensorflow objection Detection API在数据集上训练ssd来识别车辆的前后方向。<br>
# 车辆数据集
数据集的每个图像由安装在汽车仪表板上的摄像头捕获。dlib Vehicle数据集中车辆后视图及其相关边界框的示例如下:<br>
![](https://github.com/czwinner/DeepLearning/blob/master/ssds/pictures/1.png)
# 目录结构
config中的dlib_front_rear_config.py进行项目的配置。<br>
build_vehicle_records.py用于创建数据集的TensorFlow Record文件。<br>
predict.py用来进行预测和识别。<br>
dlib_front_and_rear_vehicles_v1目录包含车辆数据集。数据下载[链接](http://dlib.net/files/data/)。数据集中包含图像文件，training.xml和testing.xml。xml文件包含训练和测试集的边界框坐标和类标签。<br>
# 构建车辆数据集
在dlib_front_and_rear_vehicles_v1下创建records目录。执行python build_vehicle_records.py,查看records目录会生成下面三个文件:<br>
classes.pbtxt testing.record training.record
# 训练ssd
为了在车辆数据集上训练ssd,在[Tensorflow Object Detection Models Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)下载SSD+Inception_V2 on COCO,即ssd_inception_v2_coco_2018_01_28.tar.gz,把文件解压到dlib_front_and_rear_vehicles_v1/experiments/training目录下。<br>
接着下载配置文件，[地址](https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs)下载ssd_inception_v2_pets.config文件，修改配置文件:<br>
修改num_classes为2，因为只有两个类，分别是车辆的前和后。<br>
修改fine_tune_checkpoint
```
fine_tune_checkpoint:"/home/czwinner/ssds/dlib_front_and_rear_vehicles_v1/experiments/training/ssd_inception_v2_coco_2018_01_28/model.ckpt"
```
