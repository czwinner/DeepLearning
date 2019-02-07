# import the necessary packages
import os

image_types = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def list_images(basePath, contains=None):
    # 返回有效的文件集
    return list_files(basePath, validExts=image_types, contains=contains)


def list_files(basePath, validExts=None, contains=None):
    # 循环目录结构
    for (rootDir, dirNames, filenames) in os.walk(basePath):
        # 循环当前目录的文件名
        for filename in filenames:
            # 如果contains字符串不是none且文件名不包含提供的字符串，则忽略该文件
            if contains is not None and filename.find(contains) == -1:
                continue

            # 确定当前文件的扩展名
            ext = filename[filename.rfind("."):].lower()

            # 检查文件是否是图像应该进行处理
            if validExts is None or ext.endswith(validExts):
                # 生成图像的路径并产生它
                imagePath = os.path.join(rootDir, filename)
                yield imagePath
