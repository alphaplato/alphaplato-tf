#!/bin/python
# desc: mini demo
# why: tell how to cnn

from keras import layers
from keras import models 

def model():
    input = layers.Input(shape=(28,28,3))
    conv1 = layers.Conv2D(32,(3,3),activation='relu')(input)
    maxpool1 = layers.MaxPool2D((2,2))(conv1)
    conv2 = layers.Conv2D(64,(3,3),activation='relu')(maxpool1)
    maxpool2 = layers.MaxPool2D((2,2))(conv2)
    out = layers.Conv2D(64,(3,3),activation='relu')(maxpool2) 
    return models.Model(input,out)

out = model()
print(out.summary())

