# import the necessary packages
from keras.callbacks import BaseLogger
import matplotlib.pyplot as plt
import numpy as np
import json
import os


class TrainingMonitor(BaseLogger):
    def __init__(self, figPath, jsonPath=None, startAt=0):
        #存储图形的输出路径,JSON序列化文件的路径以及起始epoch
        super(TrainingMonitor, self).__init__()
        self.figPath = figPath  # 输出图像的路径，显示随时间的损失和准确率
        self.jsonPath = jsonPath  # 将loss和accuracy序列化为JSON文件的路径
        self.startAt = startAt  # 使用ctrl+c训练时恢复训练的开始epoch

    def on_train_begin(self, logs={}):
        #在训练过程开始时调用一次，初始化history dictionary
        self.H = {}

        # 如果JSONPath存在，加载training history
        if self.jsonPath is not None:
            if os.path.exists(self.jsonPath):
                self.H = json.loads(open(self.jsonPath).read())

                # 查看是否提供了 a starting epoch
                if self.startAt > 0:
                    # 循环历史记录日志中的条目并修剪超过starting epoch的任何条目
                    for k in self.H.keys():
                        self.H[k] = self.H[k][:self.startAt]

    def on_epoch_end(self, epoch, logs={}):
        '''
        kreas自动提供的方法，循环遍历日志并更新整个训练过程的损失，准确率等,在训练完成时调用
        '''
        for (k, v) in logs.items():
            l = self.H.get(k, [])  #l得到的是列表
            l.append(v)
            self.H[k] = l

        # 查看训练历史记录是否应序列化文件
        if self.jsonPath is not None:
            f = open(self.jsonPath, 'w')
            f.write(json.dumps(self.H))
            f.close()

        # 确保在绘图之前已经过了至少两个epochs (epoch从零开始)
        if len(self.H['loss']) > 1:
            # plot the training loss and accuracy
            N = np.arange(0, len(self.H['loss']))
            plt.style.use('ggplot')
            plt.figure()
            plt.plot(N, self.H['loss'], label='train_loss')
            plt.plot(N, self.H['val_loss'], label='val_loss')
            plt.plot(N, self.H['acc'], label='train_acc')
            plt.plot(N, self.H['val_acc'], label='val_acc')
            plt.title('Training Loss and Accurach [Epoch {}]'.format(len(self.H['loss'])))
            plt.xlabel('Epoch #')
            plt.ylabel('Loss/Accuracy')
            plt.legend()

            # save the figure
            plt.savefig(self.figPath)
            plt.close()