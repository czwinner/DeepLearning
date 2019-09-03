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
### 下载ISIS 2018数据集
要访问数据集，首先需要在ISIC 2018网站上创建一个帐户：[ISIC2018](https://challenge.kitware.com/#phase/5abcb19a56357d0139260e53)
注册后，单击“下载训练数据集”和“下载真实数据”按钮，下载.zip档案的训练数据和相应的真实mask数据<br>
![](https://github.com/czwinner/DeepLearning/blob/master/mask_rcnn/pictures/3.png)
皮肤病变边界分割数据集包括2,594个JPEG格式的图像（11GB）和2,594个相应的PNG格式的真实mask数据（27MB）。<br>
所有mask都被编码为单通道（灰度）图像。 mask中的每个像素只有两个值中的一个：<br>
解压缩两个训练数据后，具有以下目录结构：
```
ISIC2018_Task1-2_Training_Input
ISIC2018_Task1-2_Training_Input.zip
ISIC2018_Task1_Training_GroundTruth
ISIC2018_Task1_Training_GroundTruth.zip
```
训练图像的文件名：
```
$ ls -l ISIC2018_Task1-2_Training_Input/*.jpg | head -n 5
ISIC2018_Task1-2_Training_Input/ISIC_0000000.jpg
ISIC2018_Task1-2_Training_Input/ISIC_0000001.jpg
ISIC2018_Task1-2_Training_Input/ISIC_0000003.jpg
ISIC2018_Task1-2_Training_Input/ISIC_0000004.jpg
ISIC2018_Task1-2_Training_Input/ISIC_0000006.jpg
```
mask图像具有类似的文件名结构：
```
$ ls -l ISIC2018_Task1_Training_GroundTruth/*.png | head -n 5
ISIC2018_Task1_Training_GroundTruth/ISIC_0000000_segmentation.png
ISIC2018_Task1_Training_GroundTruth/ISIC_0000001_segmentation.png
ISIC2018_Task1_Training_GroundTruth/ISIC_0000003_segmentation.png
ISIC2018_Task1_Training_GroundTruth/ISIC_0000004_segmentation.png
ISIC2018_Task1_Training_GroundTruth/ISIC_0000006_segmentation.png
```

## 训练 Mask R-CNN
### 目录结构
isic2018目录包含ISIC 2018数据集本身。<br>
lesions.py文件包含训练Mask R-CNN的代码。
mask_rcnn_coco.h5文件是一个mask R-CNN，其骨干网咯ResNet在COCO数据集上进行了预训练。我们将根据自己的分割任务对此模型进行微调。该文件的[下载地址](https://github.com/matterport/Mask_RCNN/releases)<br>
mrcnn目录包含Matterport Keras + Mask R-CNN实现。将Mask-RCNN / mrcnn目录复制到这个项目中。
### 训练
执行如下命令:
```
python lesions.py --mode train
Starting at epoch 0. LR=0.001
Epoch 1/20
2075/2075 - 1234s 595ms/step - loss: 1.1190 - val_loss: 0.8835
Epoch 2/20
2075/2075 - 1192s 575ms/step - loss: 0.8643 - val_loss: 0.8789
Epoch 3/20
2075/2075 - 1176s 567ms/step - loss: 0.8274 - val_loss: 0.7446
......
```
前20epoch训练头层，然后，在第20轮结束时，解冻所有层，并开始整个网络训练：
```
Starting at epoch 20. LR=0.0001
Epoch 21/40
2075/2075 - 1757s 847ms/step - loss: 0.4888 - val_loss: 0.5687
Epoch 22/40
2075/2075 - 1757s 847ms/step - loss: 0.4409 - val_loss: 0.6123
......
```
### Mask R-CNN预测
Mask R-CNN已经过训练，可以用它做出预测。 打开终端并执行以下命令：
```
python lesions.py --mode predict \
--image isic2018/ISIC2018_Task1-2_Training_Input/ISIC_0000000.jpg
```
以下是一些预测结果:<br>
