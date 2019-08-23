# 概述
该项目使用Tensorflow objection Detection API在数据集上训练ssd来识别车辆的前后方向。<br>
# 车辆数据集
数据集的每个图像由安装在汽车仪表板上的摄像头捕获。dlib Vehicle数据集中车辆后视图及其相关边界框的示例如下:<br>
![](https://github.com/czwinner/DeepLearning/blob/master/ssds/pictures/1.png)
# 目录结构
config中的dlib_front_rear_config.py进行项目的配置。<br>
build_vehicle_records.py用于创建数据集的TensorFlow Record文件。<br>
predict.py用来进行预测和识别。<br>
dlib_front_and_rear_vehicles_v1目录包含车辆数据集。数据下载[链接](http://dlib.net/files/data/)
