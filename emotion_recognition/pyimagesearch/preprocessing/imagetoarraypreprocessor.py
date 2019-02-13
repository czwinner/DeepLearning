from keras.preprocessing.image import img_to_array
class ImageToArrayPreprocessor:
    def __init__(self,dataFormat=None):
        #存储图像数据格式
        self.dataFormat=dataFormat
    def preprocess(self,image):
        #应用keras程序重新排列图像尺寸
        return img_to_array(image,data_format=self.dataFormat)