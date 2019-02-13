<h3>验证码数字自动识别</h3>
download_images.py文件从网站上抓取验证码数字图片。<br/>
downloads目录将存储从网站下载的原始验证码图片文件。 <br/>
我们需要从每个图像中提取数字并手动标记每个数字 - 这将由annotate.py完成。<br/>
标记好的数字图像存放在dataset目录中。<br/>
train_model.py在标记的数字上训练LeNet，在output目录中，我们将存储我们训练LeNet架构。<br/>
而test_model.py将LeNet应用于验证图像本身。<br/>
<h4>随机抽取的图片显示结果

![](https://github.com/czwinner/DeepLearning/blob/master/captcha_breaker/results/result_01.png)
![](https://github.com/czwinner/DeepLearning/blob/master/captcha_breaker/results/result_02.png)
![](https://github.com/czwinner/DeepLearning/blob/master/captcha_breaker/results/result_03.png)
![](https://github.com/czwinner/DeepLearning/blob/master/captcha_breaker/results/result_04.png)
