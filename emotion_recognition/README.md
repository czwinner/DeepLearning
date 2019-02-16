<h3>The FER13 数据集
<h4>&emsp;&emsp;这个面部表情数据集称为FER13数据集，数据集由28,709个图像组成，每个图像是48x48灰度图像。我们的目标是将每张脸上表达的情绪分为七个不同的类别：愤怒，厌恶，恐惧，快乐，悲伤，惊讶和中立。下载数据集后，您将找到名为fer2013.csv的文件。<br/>
&emsp;&emsp;有三列：<br/>
&emsp;&emsp;情绪：类标签。<br/>
&emsp;&emsp;像素：表示面部本身的48x48 = 2304灰度像素的展平列表。<br/>
&emsp;&emsp;用法：图像是用于Training，PrivateTest（验证）还是PublicTest（测试）。<h4/>
<h3>构建FER13数据集
<h4>&emsp;&emsp;创建一个名为config的目录。 在配置内部，创建一个名为emotion_config.py的文件 - 这个文件是我们存储任何配置变量的地方，包括输入数据集的路径，输出HDF5文件和批量大小。<br/>
&emsp;&emsp;build_dataset.py将负责读取fer2013.csv数据集文件并输出设置一组HDF5文件; 分别用于每个训练，验证和测试集。<br/> 
&emsp;&emsp;在FER13中总共有七个类别：愤怒，厌恶，恐惧，快乐，悲伤，惊讶和中立。 然而，“厌恶”类存在严重的类不平衡，因为它只有113个图像样本（其余的每个类有超过1,000个图像）。我们将“厌恶”和“愤怒”合并为一个单独的类（因为情感在视觉上相似），从而将FER13变成了6类问题。<h4/>
<h3>定义emotionvggnet网络结构
<h4>&emsp;&emsp;pyimagesearch中的nn里的conv目录中定义emotionvggnet.py<h4/>
<h3>训练我们的面部表情识别器
<h4>&emsp;&emsp;train_recognizer.py训练我们的网络，训练75轮。<h4/>
  
![](https://github.com/czwinner/DeepLearning/blob/master/emotion_recognition/datasets/fer2013/output/vggnet_emotion.png)
<h3>评估网络
<h4>&emsp;&emsp;test_recognizer.py评估网络
