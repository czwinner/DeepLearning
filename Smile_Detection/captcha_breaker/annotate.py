from imutils import paths
import argparse
import imutils
import cv2
import os
#命令行解析
ap=argparse.ArgumentParser()
ap.add_argument("-i","--input",required=True,
                help="path to input directory of images")
ap.add_argument("-a","--annot",required=True,
                help="path to output directory of annotations")
args=vars(ap.parse_args())
#抓取图像路径然后初始化字符计数字典
imagePaths=list(paths.list_images(args["input"]))
counts={}
#循环图像路径
for (i,imagePath) in enumerate(imagePaths):
    #显示更新
    print("[INFO] processing image {} / {}".format(i+1,len(imagePaths)))
    try:
        #加载图像并转换为灰度，然后填充图像确保保留图像边框上捕获的数字
        image=cv2.imread(imagePath)
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        gray=cv2.copyMakeBorder(gray,8,8,8,8,cv2.BORDER_REPLICATE)
        #阈值图像以显示数字
        thresh=cv2.threshold(gray,0,255,
                             cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        #找到图像中的轮廓，仅保留四个最大的轮廓
        cnts=cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,
                              cv2.CHAIN_APPROX_SIMPLE)
        cnts=cnts[0] if imutils.is_cv2() else cnts[1]
        cnts=sorted(cnts,key=cv2.contourArea,reverse=True)[:4]
        #循环轮廓
        for c in cnts:
            #计算轮廓的边界框然后提取数字
            (x,y,w,h)=cv2.boundingRect(c)
            roi=gray[y-5:y+h+5,x-5:x+w+5]
            cv2.imshow("ROI",imutils.resize(roi,width=28))
            key=cv2.waitKey(0)
            #如果" ' " 键按下，则忽略这个字符
            if key==ord("'"):
                print("[INFO] ignoring character")
                continue
            #抓取按下的键并构造输出目录的路径
            key=chr(key).upper()
            dirPath=os.path.sep.join([args["annot"],key])
            #如果输出目录不存在，创建它
            if not os.path.exists(dirPath):
                os.makedirs(dirPath)
            #将带标签的字符写入文件
            count=counts.get(key,1)
            p=os.path.sep.join([dirPath,"{}.png".format(
                str(count).zfill(6))])
            cv2.imwrite(p,roi)
            #递增当前键的计数
            counts[key]=count+1
    except KeyboardInterrupt:
        print("[INFO] manually leaving script")
        break
    #次图像发生未知错误
    except:
        print("[INFO] skipping image...")