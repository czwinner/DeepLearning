from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import img_to_array
from keras.utils import np_utils
from pyimagesearch.nn.conv.lenet import LeNet
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import imutils
import cv2
import os

# 命令行解析
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
    help="path to input dataset of faces")
ap.add_argument("-m", "--model", required=True,
    help="path to output model")
args = vars(ap.parse_args())

# 初始化数据和标签列表
data = []
labels = []

# 循环输入图像
for imagePath in sorted(list(paths.list_images(args["dataset"]))):
    # 加载图片,预处理, 存储进数据列表
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = imutils.resize(image, width=28)
    image = img_to_array(image)
    data.append(image)

    # 加载类标签 更新标签列表
    label = imagePath.split(os.path.sep)[-3]
    label = "smiling" if label == "positives" else "not_smiling"
    labels.append(label)

# 把像素强度转到 [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# 标签从数字转为向量
le = LabelEncoder().fit(labels)
labels = le.transform(labels)
labels = np_utils.to_categorical(labels, 2)

# 说明标记数据的偏差
classTotals = labels.sum(axis=0)
classWeight = classTotals.max() / classTotals

# 使用80％的数据进行训练并将剩余的20％用于测试，将数据划分为训练和测试分组
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20,
    stratify=labels, random_state=42)

# 初始化模型
print("[INFO] compiling model...")
model = LeNet.build(width=28, height=28, depth=1, classes=2)
model.compile(loss="binary_crossentropy", optimizer="adam",
    metrics=["accuracy"])

# 训练
print("[INFO] training network...")
H = model.fit(trainX, trainY, validation_data=(testX, testY),
    class_weight=classWeight, batch_size=64, epochs=15, verbose=1)

# 评估模型
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=64)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1),
    target_names=le.classes_))

# 保存模型
print("[INFO] serializing network...")
model.save(args["model"])

# 画图
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 15), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 15), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 15), H.history["acc"], label="acc")
plt.plot(np.arange(0, 15), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()
