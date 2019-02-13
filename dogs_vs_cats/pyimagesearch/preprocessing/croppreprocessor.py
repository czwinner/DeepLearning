import numpy as np
import cv2
class CropPreprocessor:
    def __init__(self,width,height,horiz=True,inter=cv2.INTER_AREA):
        #存储目标图像的宽度，高度，是否水平翻转，插值方法
        self.width=width
        self.height=height
        self.horiz=horiz
        self.inter=inter
    def preprocess(self,image):
        #初始化剪裁列表
        crops=[]
        #抓取图像的宽度和高度，然后使用这些尺寸来定义图像的角
        (h,w)=image.shape[:2]
        coords=[
            [0,0,self.width,self.height],
            [w-self.width,0,w,self.height],
            [w-self.width,h-self.height,w,h],
            [0,h-self.height,self.width,h]]
        #计算图像的剪裁中心
        dW=int(0.5*(w-self.width))
        dH=int(0.5*(h-self.height))
        coords.append([dW,dH,w-dW,h-dH])
        #循环坐标，提取每个剪裁并将每个剪裁的大小调整为固定大小
        for (startX,startY,endX,endY) in coords:
            crop=image[startY:endY,startX:endX]
            crop=cv2.resize(crop,(self.width,self.height),
                            interpolation=self.inter)
            crops.append(crop)
        #检查是否应该采取水平翻转
        if self.horiz:
            #计算每个剪裁的水平镜像翻转
            mirrors=[cv2.flip(c,1) for c in crops]
            crops.extend(mirrors)
        #返回剪裁
        return np.array(crops)