from pyimagesearch.nn.conv.resnet import ResNet
from keras.utils import plot_model
model = ResNet.build(32, 32, 3, 10, (9, 9, 9),(64, 64, 128, 256), reg=0.0005)
plot_model(model, to_file="resnet.png", show_shapes=True)