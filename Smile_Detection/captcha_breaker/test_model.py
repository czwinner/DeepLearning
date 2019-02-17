from keras.preprocessing.image import img_to_array
from keras.models import load_model
from pyimagesearch.utils.captchahelper import preprocess
from imutils import contours
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
#命令行解析
ap=argparse.ArgumentParser()
ap.add_argument("-i","--input",required=True,
                help="path to input directory of images")
ap.add_argument("-m","--model",required=True,
                help="path to input model")
args=vars(ap.parse_args())
#加载模型
print("[INFO] loading pre-trained network...")
model=load_model(args["model"])
#随机抽一些图片
imagePaths=list(paths.list_images(args["input"]))
imagePaths=np.random.choice(imagePaths,size=(10,),replace=False)
#循环图像
for imagePath in imagePaths:
    #加载图像并转为灰度图，然后填充图像以确保仅保留捕获图像边框的数字
    image=cv2.imread(imagePath)
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    gray=cv2.copyMakeBorder(gray,20,20,20,20,cv2.BORDER_REPLICATE)
    #阈值图像以显示数字
    thresh=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    #找到图像中的轮廓，仅保留四个最大的轮廓，然后从左到右对它们进行排序
    cnts=cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts=cnts[0] if imutils.is_cv2() else cnts[1]
    cnts=sorted(cnts,key=cv2.contourArea,reverse=True) [:4]
    cnts=contours.sort_contours(cnts)[0]
    #将输出图像初始化具有3个通道的灰度图像，初始化预测值列表
    output=cv2.merge([gray]*3)
    predictions=[]
    #循环轮廓
    for c in cnts:
        #计算轮廓的边界框然后提取数字
        (x,y,w,h)=cv2.boundingRect(c)
        roi=gray[y-5:y+h+5,x-5:x+w+5]
        #预处理ROI并对其进行分类啊然后对其进行分类
        roi=preprocess(roi,28,28)
        roi=np.expand_dims(img_to_array(roi),axis=0) / 255.0
        pred=model.predict(roi).argmax(axis=1)[0]+1
        predictions.append(str(pred))
        #在输出图像上绘制预测
        cv2.rectangle(output,(x-2,y-2),(x+w+4,y+h+4),(0,255,0),1)
        cv2.putText(output,str(pred),(x-5,y-5),cv2.FONT_HERSHEY_COMPLEX,0.55,
                    (0,255,0),2)
    #显示输出图像
    print("[INFO] captcha: {}".format("".join(predictions)))
    cv2.imshow("Output",output)
    cv2.waitKey()