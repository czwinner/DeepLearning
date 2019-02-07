from keras.applications import ResNet50
from keras.utils import plot_model
model=ResNet50(weights="imagenet")
plot_model(model,to_file="resnet50.png",show_shapes=True)