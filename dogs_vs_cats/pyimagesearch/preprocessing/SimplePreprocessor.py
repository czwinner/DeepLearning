import cv2
class SimplePreprocessor:
    def __init__(self,width,height,inter=cv2.INTER_AREA):
        #存储目标图像的宽度、高度和插值方法
        self.width=width
        self.height=height
        self.inter=inter
    def preprocess(self,image):
        #调整图像到固定的尺寸，忽略纵横比
        return cv2.resize(image,(self.width,self.height),interpolation=self.inter)