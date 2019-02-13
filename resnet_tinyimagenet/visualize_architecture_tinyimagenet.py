from pyimagesearch.nn.conv.resnet import ResNet
from keras.utils import plot_model
model=ResNet.build(64,64,3,200,(3,4,6),
                   (64,128,256,512),reg=0.0005,dataset="tiny_imagenet")
plot_model(model, to_file="resnet_tinyimagenet.png", show_shapes=True)