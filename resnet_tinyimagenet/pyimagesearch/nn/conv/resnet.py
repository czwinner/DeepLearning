from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import AveragePooling2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model
from keras.layers import add
from keras.regularizers import l2
from keras import backend as K
class ResNet:
    @staticmethod
    def residual_module(data,K,stride,chanDim,red=False,
                        reg=0.0001,bnEps=2e-5,bnMom=0.9):
        #将ResNet模块的shortcut分支初始化为输入数据
        shortcut=data
        # ResNet模块的第一个块是1x1 CONV
        bn1=BatchNormalization(axis=chanDim,epsilon=bnEps,momentum=bnMom)(data)
        act1=Activation("relu")(bn1)
        conv1=Conv2D(int(K*0.25),(1,1),use_bias=False,kernel_regularizer=l2(reg))(act1)
        # ResNet模块的第二个块是3x3 CONV
        bn2=BatchNormalization(axis=chanDim,epsilon=bnEps,momentum=bnMom)(conv1)
        act2=Activation("relu")(bn2)
        conv2=Conv2D(int(K*0.25),(3,3),strides=stride,padding="same",use_bias=False,
                     kernel_regularizer=l2(reg))(act2)
        # ResNet模块的第三个块是另一组1x1 CONV
        bn3 = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(conv2)
        act3 = Activation("relu")(bn3)
        conv3 = Conv2D(K, (1, 1),  use_bias=False,
                       kernel_regularizer=l2(reg))(act3)
        #如果要减小空间大小，将CONV图层应用于shortcut
        if red:
            shortcut=Conv2D(K,(1,1),strides=stride,use_bias=False,
                            kernel_regularizer=l2(reg))(act1)
        # add together the shortcut and the final CONV
        x=add([conv3,shortcut])
        # return the addition as the output of the ResNet module
        return x
    @staticmethod
    def build(width,height,depth,classes,stages,filters,
              reg=0.0001,bnEps=2e-5,bnMom=0.9,dataset="cifar"):
        #初始化input shape
        inputShape=(height,width,depth)
        chanDim=-1
        # if "channel first",update the input shape
        if K.image_data_format()=="channel_first":
            inputShape=(depth,height,width)
            chanDim=1
        # set the input and apply BN
        inputs=Input(shape=inputShape)
        x=BatchNormalization(axis=chanDim,epsilon=bnEps,
                             momentum=bnMom)(inputs)
        if dataset=="cifar":
            #apply a single CONV layer
            x=Conv2D(filters[0],(3,3),use_bias=False,
                     padding="same",kernel_regularizer=l2(reg))(x)
        elif dataset=="tiny_imagenet":
            #apply CONV=>BN=>ACT=>POOL to reduce spatial size
            x=Conv2D(filters[0],(5,5),use_bias=False,
                     padding="same",kernel_regularizer=l2(reg))(x)
            x=BatchNormalization(axis=chanDim,epsilon=bnEps,
                                 momentum=bnMom)(x)
            x=Activation("relu")(x)
            x=ZeroPadding2D((1,1))(x)
            x=MaxPooling2D((3,3),strides=(2,2))(x)
        # loop over the number of stages
        for i in range(0,len(stages)):
            #初始化stride,apply a residual module to reduce
            # the spatial size of the input volume
            stride=(1,1) if i==0 else (2,2)
            x=ResNet.residual_module(x,filters[i+1],stride,
                                     chanDim,red=True,bnEps=bnEps,bnMom=bnMom)
            # loop over the number of layers in the stage
            for j in range(0,stages[i]-1):
                #apply a ResNet module
                x=ResNet.residual_module(x,filters[i+1],
                                         (1,1),chanDim,bnEps=bnEps,bnMom=bnMom)
        # apply BN=>ACT=>POOL
        x=BatchNormalization(axis=chanDim,epsilon=bnEps,momentum=bnMom)(x)
        x=Activation("relu")(x)
        x=AveragePooling2D((8,8))(x)
        # softmax classifier
        x=Flatten()(x)
        x=Dense(classes,kernel_regularizer=l2(reg))(x)
        x=Activation("softmax")(x)
        # create the model
        model=Model(inputs,x,name="resnet")
        # return the network architecture
        return model