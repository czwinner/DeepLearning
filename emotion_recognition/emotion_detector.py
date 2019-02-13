from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2
#命令行解析
ap=argparse.ArgumentParser()
ap.add_argument("-c","--cascade",required=True,
                help="path to where the face cascade resides")
ap.add_argument("-m","--model",required=True,
                help="path to pre-trained emotion detector CNN")
ap.add_argument("-v","--video",help="path to the (optional) video file")
args=vars(ap.parse_args())
#加载 face detector cascade,model，定义表情标签
detector=cv2.CascadeClassifier(args["cascade"])
model=load_model(args["model"])
EMOTIONS=["angry","scared","happy","sad","surprised","neutral"]
#如果video路径未提供，则抓取webcam引用
if not args.get("video",False):
    camera=cv2.VideoCapture(1)
#否则加载video
else:
    camera=cv2.VideoCapture(args["video"])
# keep looping
while True:
    #抓取当前帧
    (grabbed,frame)=camera.read()
    #当video到尾部则退出
    if args.get("video") and not grabbed:
        break
    #改变帧大小，转为灰度
    frame=imutils.resize(frame,width=300)
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #初始化一个canvas （显示每个类别的概率），然后复制当前帧
    canvas=np.zeros((220,300,3),dtype="uint8")
    frameClone=frame.copy()
    #检测输入帧
    rects=detector.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,
                                    minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
    #确定至少一个面部被检测到
    if len(rects)>0:
        #确定最大的脸部区域
        rect=sorted(rects,reverse=True,
                    key=lambda x:(x[2]-x[0])*(x[3]-x[1]))[0]
        (fX,fY,fW,fH)=rect
        #提取图像的ROI,然后预处理
        roi=gray[fY:fY+fH,fX:fX+fW]
        roi=cv2.resize(roi,(48,48))
        roi=roi.astype("float") /255.0
        roi=img_to_array(roi)
        roi=np.expand_dims(roi,axis=0)
        #在ROI上做预测，然后得出类别
        preds=model.predict(roi)[0]
        label=EMOTIONS[preds.argmax()]
        #循环labels+probabilities and draw them
        for (i,(emotion,prob)) in enumerate(zip(EMOTIONS,preds)):
            #构造label text
            text="{} : {:.2f} %".format(emotion,prob*100)
            #在canvas上画出label+probability
            w=int(prob*300)
            cv2.rectangle(canvas,(5,(i*35)+5),
                          (w,(i*35)+35),(0,0,225),-1)
            cv2.putText(canvas,text,(10,(i*35)+23),
                        cv2.FONT_HERSHEY_SIMPLEX,0.45,(255,255,255),2)
            #画出label on the frame
            cv2.putText(frameClone,label,(fX,fY-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,0,255),2)
            cv2.rectangle(frameClone,(fX,fY),(fX+fW,fY+fH),(0,0,255),2)
    #画出类别+概率
    cv2.imshow("Face",frameClone)
    cv2.imshow("Probabilities",canvas)
    #如果按'q'，退出循环
    if cv2.waitKey(1) & 0xFF==ord("q"):
        break
#clean up the camera and close open windows
camera.release()
cv2.destroyAllWindows()
