# Tensorflow Object Detection 训练faster_rcnn
## 数据集LiSA Traffic Signs
该数据集又47中不同的美国交通标志组成，包括停车标志，任性横道标志等。在6610帧上总共有7855个注释。道路标志的分辨率从6x6到167x168像素不等。有些是彩色，有些事灰度图像。<br>
本项目主要对三个交通标志进行训练:停车标志，人行横道线和交通信号灯。LISA数据集[下载地址](http://cvrr.ucsd.edu/LISA/lisa-traffic-sign-dataset.html)。解压后放在lisa目录中。<br>
## 安装Tensorflow object Detection API
### 环境
* Ubuntu 16.04
* CUDA 9
* CUDNN7
* Tensorflow1.12
* Python3
安装和配置Tensorflow Object Detection API 可以参考[官方指南](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md)
## 项目目录结构
TFOD API被克隆到主目录，我的TFOD API根路径/home/czwinner/models/research<br>
lisa_config.py文件存储配置信息，如文件路径，类标签等。<br>
build_lisa_records.py接受输入图像，创建训练和测试集，创建Tensorflow record文件<br>
网络训练后可以使用predict.py和predict_video.py应用于图像和视频。<br>
在lisa目录中创建records和experiments目录。records目录将会存储training.record,testing.record和classes.pbtxt<br>
experiemnts目录下包含training和exported_model目录。training目录存储微调预训练模型。exported_model在训练后存储导出的pb模型。<br>
执行python build_lisa_records.py后查看lisa/records可以看到
```linux
ls lisa/records/
classes.pbtxt testing.record training.record
```
