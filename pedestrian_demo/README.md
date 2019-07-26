## 安装Tensorflow_Object_Detection API
在[install](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md)
上查看pip install先决条件<br>
![](https://github.com/czwinner/DeepLearning/blob/master/pedestrian_demo/pictures/%E5%AE%89%E8%A3%85%E5%85%88%E5%86%B3%E6%9D%A1%E4%BB%B6.png)

git clone到本地
自己克隆到
```git
git clone https://github.com/tensorflow/models.git D:/tensorflow/models
```
进入 D:\tensorflow\models\research目录
下载protoc软件3.4版本，[网址](https://github.com/protocolbuffers/protobuf/releases/tag/v3.4.0)下载proto-3.4.0-win32.zip,解压保存在D:\proto-3.4.0-win32
在D:\tensorflow\models\research目录执行
```
D:\proto-3.4.0-win32\bin\protoc object_detection\protos\*.proto --python_out=.
```
在python安装环境的lib\site-packages中，自己用的是anaconda,目录是anaconda3\lib\site-packages中新建tensorflow_model.pth文件，加入
```
D:\tensorflow\models\research\slim
D:\tensorflow\research\object_detection
D:\tensorflow\models\research
```
目的是执行python时可以用到clone下来的object api<br>
最后执行python object_detection/builders/model_builder_test.py
出现下图说明安装成功！
![](https://github.com/czwinner/DeepLearning/blob/master/pedestrian_demo/pictures/%E6%B5%8B%E8%AF%95%E6%88%90%E5%8A%9F.png)

## 行人数据准备
[视频数据](http://www.robots.ox.ac.uk/ActiveVision/Research/Projects/2009bbenfold_headpose/Datasets/TownCentreXVID.avi)
[标注数据](http://www.robots.ox.ac.uk/ActiveVision/Research/Projects/2009bbenfold_headpose/Datasets/TownCentre-groundtruth.top)
下载到D:\images\video下
执行extract_towncentre.py读取视频数据，视频总共7500多帧，前3600帧作为训练集
