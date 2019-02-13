import argparse
import requests
import time
import os
#命令行解析
ap=argparse.ArgumentParser()
ap.add_argument("-o","--output",required=True,
                help="path to output directory of images")
ap.add_argument("-n","--num-images",type=int,default=500,
                help="# of images to download")
args=vars(ap.parse_args())
#初始化包含我们将要下载的验证码图像的URL以及到目前为止下载的图像总数
url="https://www.e-zpassny.com/vector/jcaptcha.do"
total=0
#循环下载的图片数量
for i in range(0,args["num_images"]):
    try:
        #抓取一个新的验证码图像
        r=requests.get(url,timeout=60)
        #把图像保存到磁盘
        p=os.path.sep.join([args["output"],"{}.jpg".format(
            str(total).zfill(5))])
        f=open(p,"wb")
        f.write(r.content)
        f.close()
        #更新计数
        print("[iNFO] download: {}".format(p))
        total+=1
    #处理在下载过程中是否跑出任何异常
    except:
        print("[INFO] error downloading image...")
    #插入一个小睡眠
    time.sleep(0.1)