
# import the necessary packages
import numpy as np
import cv2
import sys


def rotate_bound(image, angle):
    # 抓取图像的尺寸，然后确定中心
    (h, w) = image.shape[:2]
    (cX, cY) = (w / 2, h / 2)

    #抓住旋转矩阵（应用角度的负值顺时针旋转），然后抓住正弦和余弦（即矩阵的旋转分量）
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # 计算图像的新边界尺寸
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # 调整旋转矩阵以考虑平移
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # 执行实际旋转并返回图像
    return cv2.warpAffine(image, M, (nW, nH))
