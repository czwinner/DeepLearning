# 验证码数字识别<br/>
*  我们将要下载的图像数据集是一组验证码图像，用于防止机器人自动注册或登录到某个网站。<br/>
*  downloads目录将存储从E-ZPass网站下载的原始captcha .jpg文件。 在output目录中，我们将存储LeNet架构。
download_images.py将负责实际下载示例验证码并将其保存到磁盘。<br/>
*  一旦我们下载了一组验证码，我们就需要从每个图像中提取数字并手动标记每个数字 - 这将由annotate.py完成。<br/>
*  dataset目录是我们存储标记数字的地方，我们将手工标记这些数字。<br/>
*  train_model.py将在标记的数字上训练LeNet，而test_model.py将LeNet应用于验证图像本身。<br/>
## 结果显示<br/>
![](https://github.com/czwinner/DeepLearning/blob/master/captcha_breaker/results/result_01.png)
![](https://github.com/czwinner/DeepLearning/blob/master/captcha_breaker/results/result_02.png)
![](https://github.com/czwinner/DeepLearning/blob/master/captcha_breaker/results/result_03.png)
![](https://github.com/czwinner/DeepLearning/blob/master/captcha_breaker/results/result_04.png)
