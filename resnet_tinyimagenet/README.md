<h3>数据集<br/>
&emsp;&emsp;Tiny ImageNet数据集实际上是完整ImageNet数据集的一个子集，包括200个不同的类，包括从埃及猫到排球到柠檬的所有类别。特定有200个类别。每个类别包括500个训练图像，50个验证图像和50个测试图像。因此我们将使用部分训练集来构建我们自己的测试集，以便我们评估分类算法的性能。Tiny ImageNet数据集中的所有图像都已调整为64x64像素并中心裁剪。<br/>
&emsp;&emsp;然后我们有train目录，以字母n开头，后跟一系列数字。 这些子目录是WordNet ID，简称为“同义词集”或“synsets”。 每个WordNet ID映射到特定的单词/对象。给定WordNet子目录中的每个图像都包含该对象的示例。我们可以通过解析words.txt文件来查找WordNet ID的人类可读标签， 这只是一个制表符分隔文件，在第一列中带有WordNet ID且人类可读 第二列中的单词/对象。wnids.txt文件列出了200个WordNet ID（每行一个） 在ImageNet数据集中。val目录存储我们的验证集。 在val目录中，您将找到images子目录和名为val_annotations.txt的文件。 val_annotations.txt为val目录中的每个映像提供WordNet ID<br/>
<h3>训练网络<br/>
&emsp;&emsp;pyimagesearch中conv模块中的resnet.py定义了resnet网络结构。<br/>
&emsp;&emsp;build_tiny_imagenet.py生成train,val和test的HDF5文件。<br/>
&emsp;&emsp;train_decay.py基于学习率衰减训练resnet网络，训练75个epoch <br/>

![](https://github.com/czwinner/DeepLearning/blob/master/resnet_tinyimagenet/output/2035.png)
<h3>评估网络<br/>
&emsp;&emsp;rany_accuracy.py评估准确率。
  
![](https://github.com/czwinner/DeepLearning/blob/master/resnet_tinyimagenet/output/%E5%B1%8F%E5%B9%95%E5%BF%AB%E7%85%A7%202019-02-13%20%E4%B8%8A%E5%8D%889.03.23.png)  
rank-1和rank-5准确率分别为50%和75%左右
