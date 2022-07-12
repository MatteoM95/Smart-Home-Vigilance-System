import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow import keras


def set_model(model_name, n_classes, input_shape, alpha): 
        
    strides=[2,1]
    units = n_classes

    if model_name == "DS-CNN":

        model = keras.Sequential([
            keras.layers.Conv2D(filters=int(alpha*256), kernel_size=[3, 3], strides=strides, use_bias=False, input_shape=input_shape),
            keras.layers.BatchNormalization(momentum=0.1),
            keras.layers.ReLU(),

            keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], use_bias=False),
            keras.layers.Conv2D(filters=int(alpha*256), kernel_size=[1, 1], strides=[1, 1], use_bias=False),
            keras.layers.BatchNormalization(momentum=0.1),
            keras.layers.ReLU(),

            keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], use_bias=False),
            keras.layers.Conv2D(filters=int(alpha*256), kernel_size=[1, 1], strides=[1, 1], use_bias=False),
            keras.layers.BatchNormalization(momentum=0.1),
            keras.layers.ReLU(),

            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dense(units=units)
            ])
            
    elif model_name == "MobileNet":
        model = keras.Sequential([
            keras.layers.Conv2D(filters=int(alpha*32), kernel_size=[3, 3], strides=strides, use_bias=False, input_shape=input_shape),
            keras.layers.BatchNormalization(momentum=0.1),
            keras.layers.ReLU(),

            keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding="same", use_bias=False),
            keras.layers.BatchNormalization(momentum=0.1),
            keras.layers.ReLU(),
            keras.layers.Conv2D(filters=int(alpha*64), kernel_size=[1, 1], strides=[1, 1], padding="same", use_bias=False),
            keras.layers.BatchNormalization(momentum=0.1),
            keras.layers.ReLU(),

            keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[2, 2], padding="same", use_bias=False),
            keras.layers.BatchNormalization(momentum=0.1),
            keras.layers.ReLU(),
            keras.layers.Conv2D(filters=int(alpha*128), kernel_size=[1, 1], strides=[1, 1], padding="same", use_bias=False),
            keras.layers.BatchNormalization(momentum=0.1),
            keras.layers.ReLU(),

            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dense(units=units)
        ])
    
    elif model_name == "MusicTaggerCNN":
        model = keras.Sequential([
            keras.layers.InputLayer(input_shape=input_shape),
            keras.layers.BatchNormalization(axis=2),

            keras.layers.Conv2D(filters=int(alpha*64), kernel_size=3, strides=3, padding="same", use_bias=False),
            keras.layers.BatchNormalization(axis=1),
            keras.layers.ELU(),
            keras.layers.MaxPool2D(pool_size=(2,4), padding="same"),

            keras.layers.Conv2D(filters=int(alpha*128), kernel_size=3, strides=3, padding="same", use_bias=False),
            keras.layers.BatchNormalization(axis=1),
            keras.layers.ELU(),
            keras.layers.MaxPool2D(pool_size=(2,4), padding="same"),

            keras.layers.Conv2D(filters=int(alpha*128), kernel_size=3, strides=3, padding="same", use_bias=False),
            keras.layers.BatchNormalization(axis=1),
            keras.layers.ELU(),
            keras.layers.MaxPool2D(pool_size=(2,4), padding="same"),
            
            keras.layers.Conv2D(filters=int(alpha*128), kernel_size=3, strides=3, padding="same", use_bias=False),
            keras.layers.BatchNormalization(axis=1),
            keras.layers.ELU(),
            keras.layers.MaxPool2D(pool_size=(3,5), padding="same"),
            
            keras.layers.Conv2D(filters=int(alpha*64), kernel_size=3, strides=3, padding="same", use_bias=False),
            keras.layers.BatchNormalization(axis=1),
            keras.layers.ELU(),
            keras.layers.MaxPool2D(pool_size=(4,4), padding="same"),

            keras.layers.Flatten(),
            keras.layers.Dense(units=units)
        ])

    elif model_name == "SimpleNet":
        model = keras.Sequential([
            keras.layers.Conv2D(filters=int(alpha*32), kernel_size=[3, 3], strides=strides, use_bias=False, input_shape=input_shape),
            keras.layers.ReLU(),
            keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding="same"),

            keras.layers.Conv2D(filters=int(alpha*64), kernel_size=[3, 3], strides=[1, 1], padding="same", use_bias=False),
            keras.layers.ReLU(),
            keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding="same"),

            keras.layers.Conv2D(filters=int(alpha*128), kernel_size=[3, 3], strides=[1, 1], padding="same", use_bias=False),
            keras.layers.ReLU(),
            keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding="same"),

            keras.layers.Conv2D(filters=int(alpha*256), kernel_size=[3, 3], strides=[1, 1], padding="same", use_bias=False),
            keras.layers.ReLU(),
            keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding="same"),

            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dense(units=units)
        ])
        
    elif model_name == "VGGish":
        model = keras.Sequential([
            keras.layers.Conv2D(filters=int(alpha*64), kernel_size=(3,3), strides=(1,1), padding="same", activation="relu", input_shape=input_shape, name='conv1'),
            keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding="same", name='pool1'),
            
            keras.layers.Conv2D(filters=int(alpha*128), kernel_size=(3,3), strides=(1,1), padding="same", activation="relu", name='conv2'),
            keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding="same", name='pool2'),
            
            keras.layers.Conv2D(filters=int(alpha*256), kernel_size=(3,3), strides=(1,1), padding="same", activation="relu", name='conv3/conv3_1'),
            keras.layers.Conv2D(filters=int(alpha*256), kernel_size=(3,3), strides=(1,1), padding="same", activation="relu", name='conv3/conv3_2'),
            keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding="same", name='pool3'),
            
            keras.layers.Conv2D(filters=int(alpha*512), kernel_size=(3,3), strides=(1,1), padding="same", activation="relu", name='conv4/conv4_1'),
            keras.layers.Conv2D(filters=int(alpha*512), kernel_size=(3,3), strides=(1,1), padding="same", activation="relu", name='conv4/conv4_2'),
            keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding="same", name='pool4'),
            
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dense(units=units)
        ])
    else:
        raise ValueError('{} not implemented'.format(model_name))  

    return model