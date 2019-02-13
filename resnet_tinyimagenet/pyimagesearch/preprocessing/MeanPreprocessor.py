import cv2
class MeanPreprocessor:
    def __init__(self,rMean,gMean,bMean):
        #存储训练集的Red,Green,Blu通道的平均值
        self.rMean=rMean
        self.gMean=gMean
        self.bMean=bMean
    def preprocess(self,image):
        #把图像分成Red,Green,Blue通道
        (B,G,R)=cv2.split(image.astype("float32"))
        #减去每个通道的平均值
        R-=self.rMean
        G-=self.gMean
        B-=self.bMean
        #将通道合并在一起并返回图像
        return cv2.merge([B,G,R])