<h2>纠正图像方向</h2>  
&emsp;&emsp;本项目应用迁移学习（特征提取）来自动检测和校正图像的方向。<br/>  
&emsp;&emsp;1.证明了ImageNet上训练的CNN学习的滤波器在整个360度中是旋转改变的的（否则，这些特征不能用于区分图像旋转）。<br/>
&emsp;&emsp;2.通过特征提取进行学习,在预测图像方向时获得很高的精度。<br/>
<h3>数据集</h3> 
&emsp;&emsp;本项目研究的数据集是麻省理工学院发布的室内场景识别（也称为Indoor CVPR）数据集。
该数据集包含67个室内类别的房间/场景类别，包括家庭，办公室，公共场所，商店等等。<br/>
&emsp;&emsp;Indoor CVPR中的所有图像都是正确定向的。因此我们需要在Indoor CVPR中构建我们自己的数据集，并在不同的旋转下使用带标记的图像。<br/>
&emsp;&emsp;下载解压后有一个名为Images的目录，其中包含许多子目录，每个子目录都包含数据集中的特定类标签。
在indoor_cvpr中再创建两个新的子目录 -  hdf5和rotating_images。<br/>
&emsp;&emsp;下hdf5目录将存储使用预训练的卷积神经网络从输入图像中提取的特征。 为了生成我们的训练数据，创建一个生成随机旋转图像的create_dataset.py。 
这些旋转的图像将存储在rotating_images中。 从这些图像中提取的特征将存储在HDF5目录中。<br/>
<h3>构建数据集</h3>
&emsp;&emsp;我们将使用create_dataset.py脚本为输入数据集构建训练和测试集。create_dataset.py构建0，90，180，270度旋转角度图像，每个旋转角度有2500个图像。
创建了自己的数据集后，可以继续通过特征提取应用迁移学习 - 这些特征将在Logistic回归分类器中用于预测（和校正）输入图像的方向。<br/>
<h3>特征提取</h3>
&emsp;&emsp;要从我们的数据集中提取特征，将使用已在ImageNet数据集上预先训练过的VGG16网络架构。
extract_features.py进行特征提取，提取的hdf5文件在hd5目录。<br/>
<h3>训练方向校正分类器</h3>
&emsp;&emsp;训练分类器来预测图像方向使用train_model.py完成 - 提供输入HDF5数据集的路径，train_model.py负责调整Logistic回归超参数并输出模型到磁盘。
训练的模型保存在models目录中。一旦train_model.py完成执行，我们的分类器获得了92％的准确率。<br/>

