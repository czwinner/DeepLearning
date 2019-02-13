<h3>微笑检测 <br/>
<h4>&emsp;&emsp;该项目可以使用深度学习和传统的计算机视觉技术实时检测视频流中的微笑。为了完成这项任务，我们将在一个图像数据集上训练LetNet架构，这些图像包含微笑或不微笑的人的面孔。一旦我们的网络经过训练，然后创建一个单独的Python脚本 - 这个脚本将通过OpenCV的内置Haar级联检测图像中的面部人脸检测器，从图像中提取感兴趣的人脸区域（ROI），然后通过ROI通过LeNet进行微笑检测。<br/>
&emsp;&emsp;SMILES数据集由微笑或不微笑的面部图像组成。 数据集中有13,165个灰度图像，每个图像的大小为64x64像素。<br/>
&emsp;&emsp;我们首先需要在图像中定位面部并提取面部ROI然后才能通过它通过我们的网络进行检测。 使用传统的计算机视觉方法Haar级联来完成。<br/>
&emsp;&emsp;我们需要在SMILES数据集中处理的第二个问题是类不平衡。 虽然数据集中有13,165个图像，但这些示例中有9,475个没有微笑，而只有3,690个属于微笑类。 比例为2.5：1。通过在训练期间计算每个类别的权重来对抗类不平衡。<br/>
&emsp;&emsp;构建我们的微笑探测器的第一步是在SMILES数据集上训练CNN，以区分微笑与不微笑的脸。 要完成此任务，让我们创建一个名为train_model.py的新文件。
经过15轮训练，网络获得了93％的分类准确率。<br/>
&emsp;&emsp;现在我们已经训练了我们的模型，构建detect_smile.py以访问我们的网络摄像头/视频文件并对每个帧应用微笑检测。<h4/>
<h3>结果显示 <br/>
  
![](https://github.com/czwinner/DeepLearning/blob/master/Smile_Detection/results/result01.png)
![](https://github.com/czwinner/DeepLearning/blob/master/Smile_Detection/results/result02.png)
