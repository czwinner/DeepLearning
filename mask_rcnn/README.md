# 基于Mask R-CNN癌症检测
## 安装 Keras Mask R-CNN
为了实际训练我们自己定制的Mask R-CNN模型，我们将使用Matterport Keras + Mask R-CNN实现。<br>
我们需要确保安装以下Python包：<br>
```python
$ pip install numpy scipy h5py
$ pip install scikit-learn Pillow
$ pip install imgaug imutils
$ pip install beautifulsoup4
$ pip install tensorflow-gpu
$ pip install keras
$ pip install opencv-contrib-python
```
使用的是TensorFlow tensorflow-gpu的GPU版本。 <br>
下一步将克隆其官方GitHub项目页面的Keras + Mask R-CNN实现：<br>
```python
$ cd ~
$ git clone https://github.com/matterport/Mask_RCNN
$ cd Mask_RCNN
```
接下来，安装Mask R-CNN包的所有要求：<br>
```python
pip install -r requirements.txt
'''
要验证是否已成功安装Keras + Mask R-CNN软件包，请从Mask_RCNN目录中打开Python shell并尝试导入mrcnn软件包：<br>
```python
$ python
>>> import mrcnn
>>>
```
如果导入成功且没有错误，则正确安装mrcnn。<br>
