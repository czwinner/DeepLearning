

# 导包
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2

# 命令行解析
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade", required=True,
    help="path to where the face cascade resides")
ap.add_argument("-m", "--model", required=True,
    help="path to pre-trained smile detector CNN")
ap.add_argument("-v", "--video",
    help="path to the (optional) video file")
args = vars(ap.parse_args())

# 加载面部检测器级联和微笑检测器CNN
detector = cv2.CascadeClassifier(args["cascade"])
model = load_model(args["model"])

# 如果未提供视频，请使用网络摄像头
if not args.get("video", False):
    camera = cv2.VideoCapture(0)
# 否则加载视频
else:
    camera = cv2.VideoCapture(args["video"])

while True:
    # 抓住当前帧
    (grabbed, frame) = camera.read()

    # if we are viewing a video and did not a grab a frame then we have reached
    # 如果我们正在观看视频并且没有抓住一帧，那么我们已经到了视频的末尾
    if args.get("video") and not grabbed:
        break

    # 调整大小，转换为灰度，然后克隆它（所以我们可以注释它）
    frame = imutils.resize(frame, width=300)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frameClone = frame.copy()

    # 检测输入frame的面部，然后克隆frame，以便我们可以在其上绘制
    rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
        minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    # 循环检测到的边界框
    for (fX, fY, fW, fH) in rects:
        # 从灰度图像中提取面部的ROI，将其调整为28x28，然后准备ROI以进行分类
        roi = gray[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (28, 28))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        (notSmiling, smiling) = model.predict(roi)[0]
        label = "smile" if smiling > notSmiling else "not smile"

        cv2.putText(frameClone, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 0, 255), 2)
        cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH), (0, 255, 0), 2)

        cv2.imshow("Face", frameClone)

        if cv2.waitKey(0) & 0xFF == ord("q"):
            break

# 释放
camera.release()
cv2.destroyAllWindows()


