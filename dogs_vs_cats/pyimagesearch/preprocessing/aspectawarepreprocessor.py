import cv2
import imutils
class AspectAwarePreprocessor:
    def __init__(self,width,height,inter=cv2.INTER_AREA):
        #存储目标图像的宽度、高度和插值方法
        self.width=width
        self.height=height
        self.inter=inter
    def preprocess(self,image):
        #抓取图像的尺寸，初始化在裁剪时使用的增量
        (h,w)=image.shape[:2]
        dW=0
        dH=0
        #如果宽度小于高度，则沿宽度（即较小尺寸）调整大小，然后更新
        # 增量以将高度裁剪为所需尺寸
        if w<h:
            image=imutils.resize(image,width=self.width,inter=self.inter)
            dH=int((image.shape[0]-self.height)/2.0)
        #否则高度小于宽度，因此沿高度调整大小，然后更新增量以沿宽度裁剪
        else:
            image=imutils.resize(image,height=self.height,inter=self.inter)
            dW=int((image.shape[1]-self.width)/2.0)
        #既然我们的图像已经调整大小，我们需要重新抓取宽度和高度，然后进行裁剪
        (h,w)=image.shape[:2]
        image=image[dH:h-dH,dW:w-dW]
        #最后，将图像大小调整为提供的空间尺寸，以确保输出图像始终是固定大小
        return cv2.resize(image,(self.width,self.height),interpolation=self.inter)