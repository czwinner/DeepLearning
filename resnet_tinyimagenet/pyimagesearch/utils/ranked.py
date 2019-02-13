import numpy as np
def rank5_accuracy(preds,labels):
    #初始化rank-1和rank-5准确度
    rank1=0
    rank5=0
    #循环预测值和真实标签
    for (p,gt) in zip(preds,labels):
        #按索引按降序对概率进行排序，以便在列表前面进行猜测
        p=np.argsort(p)[::-1]
        #查看真实标签是否在top-5预测中
        if gt in p[:5]:
            rank5 += 1
        #查看真实标签是否是 #1预测
        if gt == p[0]:
            rank1 += 1
    #计算最终rank-1 和 rank-5的准确率
    rank1 /= float(len(labels))
    rank5 /= float(len(labels))
    #返回rank-1 和 rank-5准确率的元组
    return (rank1,rank5)