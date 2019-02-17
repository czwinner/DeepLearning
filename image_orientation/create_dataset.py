from imutils import paths
import numpy as np
import progressbar
import argparse
import imutils
import random
import cv2
import os
#命令行解析
ap=argparse.ArgumentParser()
ap.add_argument("-d","--dataset",
                help="path to input directory of images")
ap.add_argument("-o","--output",required=True,
                help="path to output directory of rotated images")
args=vars(ap.parse_args())
#抓取图像的路径(限制为10000张图像),并随机混洗
imagePaths=list(paths.list_images(args["dataset"]))[:10000]
random.shuffle(imagePaths)
#初始化字典以跟踪每个角度的数量，初始化progress bar
angles={}
widgets=["Buildng Dataset: ",progressbar.Percentage(), " ",
         progressbar.Bar()," ",progressbar.ETA()]
pbar=progressbar.ProgressBar(maxval=len(imagePaths),widgets=widgets).start()
#循环image paths
for (i,imagePath) in enumerate(imagePaths):
    #确定旋转角度，加载图像
    angle=np.random.choice([0,90,180,270])
    image=cv2.imread(imagePath)
    #如果image is None(表示从批判加载图像时出现问题，只需跳过他)
    if image is None:
        continue
    #根据所选角度旋转图像，构造基本输出目录的路径
    image=imutils.rotate_bound(image,angle)
    base=os.path.sep.join([args["output"],str(angle)])
    #如果base path不存在，创建它
    if not os.path.exists(base):
        os.mkdir(base)
    #提取图像文件扩展名，然后构造输出文件的完整路径
    ext=imagePath[imagePath.rfind("."):]
    outputPath=[base,"image_{}{}".format(
        str(angles.get(angle,0)).zfill(5),ext)]
    outputPath=os.path.sep.join(outputPath)
    #保存图像
    cv2.imwrite(outputPath,image)
    #为角度更新字典数量
    c=angles.get(angle,0)
    angles[angle]=c+1
    pbar.update(i)
#完成progress bar
pbar.finish()
#循环angles字典并显示它们的数量
for angle in sorted(angles.keys()):
    print("[INFO] angle={}:{:,}".format(angle,angles[angle]))