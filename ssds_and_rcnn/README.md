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
## 配置Faster R-CNN
在LISA数据集上训练Faster R-CNN分为四步:<br>
1. 下载预训练的Faster R-CNN<br>
2. 下载示例TFOD API配置文件，修改为指向我们的record文件<br>
3. 开始训练<br>
4. 训练完成后导出冻结模型图<br>
### 下载预训练模型
前往[tensorflow model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md),下载faster_rcnn_resnet101_coco_2018_01_28.tar.gz,下载后放入experiments/training子目录并解压。
### 下载配置文件
有了模型权重，还需要一个配置文件配置如何训练/微调网络
前往[页面](https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs)下载faster_rcnn_resnet101_pets.config保存到experiments/training,打开faster_rcnn_lisa.config，修改
```python
model {
faster_rcnn {
num_classes: 37
image_resizer {
keep_aspect_ratio_resizer {
min_dimension: 600
max_dimension: 1024
}
}
feature_extractor {
type: 'faster_rcnn_resnet101'
first_stage_features_stride: 16
}
```
中的numclasses为3<br>
```python
train_config: {
85 batch_size: 1
86 ...
87 num_steps: 50000
88 data_augmentation_options {
89 random_horizontal_flip {
90 }
91 }
92 }
```
num_steps改为50000<br>
```python
fine_tune_checkpoint:"/home/czwinner/ssds_and_rcnn/lisa/experiments/training/faster_rcnn_resnet101_coco_2018_01_28/model.ckpt"
```
接着修改train_input_reader中的input_path和label_map_path
```python
train_input_reader:{
  tf_record_input_reader{
    input_path:"/home/czwinner/ssds_and_rcnn/lisa/records/trainging.record"
                        }
    label_map_path:"/home/czwinner/ssds_and_rcnn/lisa/records/classes.pbtxt"
                    }
```
同理修改eval_input_reader中的input_path和label_map_path
## 训练Faster R-CNN
可以用以下命令开始训练
```
python object_detection/model_main.py
--pipeline_config_path=ssds_and_rcnn/lisa/experiments/training/faster_rcnn_lisa.config
--model_dir=ssds_and_rcnn/lisa/experiemtns/training
--num_train_steps=50000
--sample_1_of_n_eval_examples=1
--alsologtostderr
```
## 导出冻结模型图
模型训练好后进入models/research目录执行<br>
```
python object_detection/export_inference_graph.py
--input_type image_tensor
--pipeline_config_path /home/czwinner/ssds_and_rcnn/lisa/experiments/traing/faster_rcnn_lisa.config
--trained_checkpoint_prefix /home/czwinner/ssds_and_rcnn/lisa/experiments/trainging/model.ckpt-50000
--output_directory /home/czwinner/ssds_and_rcnn/lisa/experiments/exported_model
```
查看exported_model目录，可以看到frozen_inference_grpah.pb文件
## 在图像上测试
训练并导出模型后就可以评估准确性，执行
```
python predict.py
--model lisa/experiments/exported_model/frozen_inference_graph.pb
--labels lisa/records/classes.pbtxt
--image path/to/input/image.png
--num-classes 3
```
