<h3>数据集<h3/><br/>

&nbsp;&nbsp;Tiny ImageNet数据集实际上是完整ImageNet数据集的一个子集（因此无法使用特征提取，因为它会给网络带来不公平的优势），包括200个不同的类，包括从埃及猫到排球到柠檬的所有类别。特定有200个类别。每个类别包括500个训练图像，50个验证图像和50个测试图像。因此我们将使用部分训练集来构建我们自己的测试集，以便我们评估分类算法的性能。Tiny ImageNet数据集中的所有图像都已调整为64x64像素并中心裁剪。<br/>
&nbsp;&nbsp;然后我们有train目录，其中包含奇怪名称的子目录，以字母n开头，后跟一系列数字。 这些子目录是WordNet [43] ID，简称为“同义词集”或“synsets”。 每个WordNet ID映射到特定的单词/对象。给定WordNet子目录中的每个图像都包含该对象的示例。
&nbsp;&nbsp;我们可以通过解析words.txt文件来查找WordNet ID的人类可读标签， 这只是一个制表符分隔文件，在第一列中带有WordNet ID且人类可读 第二列中的单词/对象。wnids.txt文件列出了200个WordNet ID（每行一个） 在ImageNet数据集中。
&nbsp;&nbsp;最后，val目录存储我们的验证集。 在val目录中，您将找到images子目录和名为val_annotations.txt的文件。 val_annotations.txt为val目录中的每个映像提供WordNet ID
