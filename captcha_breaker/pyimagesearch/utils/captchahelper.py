import imutils
import cv2
def preprocess(image,width,height):
    #抓取图像的尺寸，然后初始化填充值
    (h,w)=image.shape[:2]
    #如果宽度大于高度，则沿宽度调整大小
    if w > h:
        image=imutils.resize(image,width=width)
    #否则，高度大于宽度，因此沿高度调整大小
    else:
        image=imutils.resize(image,height=height)
    #确定宽度和高度的填充值以获取目标尺寸
    padW=int((width-image.shape[1]) / 2.0)
    padH = int((height-image.shape[0]) / 2.0)
    #填充图像然后再应用一次调整大小以处理任何舍入问题
    image=cv2.copyMakeBorder(image,padH,padH,padW,padW,
                             cv2.BORDER_REPLICATE)
    image=cv2.resize(image,(width,height))
    #返回预处理的图像
    return image