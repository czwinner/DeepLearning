#导包
import h5py
import os
class HDF5DatasetWriter:
    def __init__(self,dims,outputPath,dataKey="images",bufSize=1000):
        #查看output路径是否存在
        if os.path.exists(outputPath):
            raise ValueError("The supplied 'outputPath' already"
                             "exists and cannot be overwritten.Manually delete"
                             "the file before continuing.",outputPath)
        #打开HDF5数据库写入并创建两个数据集:
        #一个存储图像/特征，另一个存储类标签
        self.db=h5py.File(outputPath,'w')
        self.data=self.db.create_dataset(dataKey,dims,dtype="float")
        self.labels=self.db.create_dataset("labels",(dims[0],),dtype="int")
        #存储缓冲区大小，然后将缓冲区本身与索引一起初始化到数据集中
        self.bufSize=bufSize
        self.buffer={"data":[], "labels": []}
        self.idx=0
    def add(self, rows, labels):
        #添加行和标签到buffer中
        self.buffer["data"].extend(rows)
        self.buffer["labels"].extend(labels)
        #检查是否需要将缓冲区刷新到磁盘
        if len(self.buffer["data"])>=self.bufSize:
            self.flush()
    def flush(self):
        #将缓冲区写入磁盘然后重置缓冲区
        i=self.idx+len(self.buffer["data"])
        self.data[self.idx:i]=self.buffer["data"]
        self.labels[self.idx:i]=self.buffer["labels"]
        self.idx=i
        self.buffer={"data":[],"labels":[]}
    def storeClassLabels(self,classLabels):
        #创建一个数据集来存储实际的类标签名称，然后存储类标签
        dt=h5py.special_dtype(vlen=str)
        labelSet=self.db.create_dataset("label_names",(len(classLabels),), dtype=dt)
        labelSet[:]=classLabels
    def close(self):
        #检查缓冲区中是否有其他要到磁盘的条目
        if len(self.buffer["data"])>0:
            self.flush()
        #关闭数据集
        self.db.close()