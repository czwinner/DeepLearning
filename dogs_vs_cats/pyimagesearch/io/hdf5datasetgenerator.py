from keras.utils import np_utils
import numpy as np
import h5py


class HDF5DatasetGenerator:
    def __init__(self,dbPath,batchSize,preprocessors=None,aug=None,
                 binarize=True,classes=2):
        #存储批量大小，预处理器和数据增强，标签是否二值化，类别总数
        self.batchSize=batchSize
        self.preprocessors=preprocessors
        self.aug=aug
        self.binarize=binarize
        self.classes=classes
        #打开HDF5数据库进行读取并确定数据库中的条目总数
        self.db=h5py.File(dbPath)
        self.numImages = self.db["labels"].shape[0]
    def generator(self, passes=np.inf):
        #初始化epoch计数
        epochs=0
        #保持无限循环，一旦达到所需epoch数，模型就会停止
        while epochs < passes:
            #循环HDF5数据集
            for i in np.arange(0,self.numImages,self.batchSize):
                #提取HDF5数据集的图像和标签
                images=self.db["images"][i:i+self.batchSize]
                labels=self.db["labels"][i:i+self.batchSize]
                #查看标签是否二值化
                if self.binarize:
                    labels=np_utils.to_categorical(labels,self.classes)
                #查看是否预处理器非空
                if self.preprocessors is not None:
                    #初始化已处理图像的列表
                    procImages=[]
                    #循环图像
                    for image in images:
                        #循环预处理器并应用于每个图像
                        for p in self.preprocessors:
                            image=p.preprocess(image)
                        #更新已处理图像列表
                        procImages.append(image)
                    #更新image数组已处理的图像
                    images=np.array(procImages)
                #查看图像增强是否存在
                if self.aug is not None:
                    (images,labels)=next(self.aug.flow(images,labels,batch_size=self.batchSize))
                #产生images和labels元组
                yield (images,labels)
            #增加epochs总数
            epochs+=1
    def close(self):
        #关闭数据集
        self.db.close()