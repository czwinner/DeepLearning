# 基于Mask R-CNN癌症检测
## 安装 Keras Mask R-CNN
为了实际训练我们自己定制的Mask R-CNN模型，我们将使用Matterport Keras + Mask R-CNN实现。<br>
我们需要确保安装以下Python包：<br>
```
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
```
$ cd ~
$ git clone https://github.com/matterport/Mask_RCNN
$ cd Mask_RCNN
```
接下来，安装Mask R-CNN包的所有要求：<br>
```
pip install -r requirements.txt
```
要验证是否已成功安装Keras + Mask R-CNN软件包，请从Mask_RCNN目录中打开Python shell并尝试导入mrcnn软件包：<br>
```
$ python
>>> import mrcnn
>>>
```
如果导入成功且没有错误，则正确安装mrcnn。<br>
## ISIS皮肤病变数据集
在本项目中使用的数据集是(ISIC)2018皮肤病变数据集。该数据集能够为癌症，特别是黑色素瘤的研究和早期检测做出贡献。早期发现，黑色素瘤的存活率超过95％。因此，早期发现癌症是关键。<br>
下图中是皮肤病变边界检测的示例。顶行包含皮肤损伤图像的示例，而底行包含其相应的掩模。我们的目标是从输入图像中正确预测那些像素方式的掩模。<br>
![](https://github.com/czwinner/DeepLearning/blob/master/mask_rcnn/pictures/%E5%B1%8F%E5%B9%95%E5%BF%AB%E7%85%A7%202019-09-03%20%E4%B8%8A%E5%8D%888.27.50.png)
