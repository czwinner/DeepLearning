<h3>猫狗识别 
 <h4> 构建HDF5数据集<br/>
&emsp;&emsp;数据集来自kaggle的Dogs_vs_Cats,包含25000张猫狗图片。train的目录 - 这个目录包含我们的实际图像。标签本身可以从检查文件名中获得。build_dogs_vs_cats.py构建HDF5数据集。<br/>
<h4>图像预处理<br/>
&emsp;&emsp;1.平均减法预处理，用于从输入图像中减去数据集中的平均红色，绿色和蓝色像素强度（这是数据标准化的一种形式）。<br/>
&emsp;&emsp;2.patch预处理，用于在训练期间从图像中随机提取MxN像素区域。<br/>
&emsp;&emsp;3.过采样预处理，用于在测试时对输入图像的五个区域（四个角+中心区域）进行采样以及相应的水平翻转（总共10个图片）。<br/>
&emsp;&emsp;平均预处理由pyimagesearch的preprocessing的MeanPreprocessor.py完成。<br/>
&emsp;&emsp;patch预处理由pyimagesearch的preprocessing的PatchPreprocessor.py完成。<br/>
&emsp;&emsp;过采样预处理由pyimagesearch的preprocessing的croppreprocessor.py完成。<br/>
&emsp;&emsp;在我们实现AlexNet架构并在Kaggle Dogs vs. Cats数据集上进行训练之前，我们首先需要定义一个类，负责从我们的HDF5数据集中生成批量图像和标签。由pyimagesearch的io中的hdf5datasetgenerator.py处理。<br/>
<h4>定义Alexnet<br/>
&emsp;&emsp;为了实现AlexNet，让我们在pyimagesearch中的nn的conv子模块中创建一个名为alexnet.py的文件。
<h4>训练Alexnet<br/>
&emsp;&emsp;定义了AlexNet架构之后，训练网络。 打开一个新文件，将其命名为train_alexnet.py。我们看到获得的验证集准确度为92.97％。<br/>

  
  
