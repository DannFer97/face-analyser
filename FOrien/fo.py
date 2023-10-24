#Librerias
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Add, Activation
from tensorflow.keras import Model


    #from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import EfficientNetV2B1
from tensorflow.keras.optimizers import Adam
model_original = EfficientNetV2B1(input_shape=(230,230,3),include_top=False,weights="imagenet")

#--------------------------
inputs = model_original.output
bn0 = BatchNormalization(scale=True)(inputs)
bn0=(bn0)
    # Initial Stage
conv1 = Conv2D(32, kernel_size=(7,7), padding='same', activation='relu', kernel_initializer='uniform')(bn0)
conv1 = Conv2D(32, kernel_size=(7,7), padding='same', activation='relu', kernel_initializer='uniform')(conv1)
bn1 = BatchNormalization(scale=True)(conv1)
max_pool1 = MaxPooling2D(pool_size=(2,2))(bn1)

    # First
conv2 = Conv2D(32, kernel_size=(5,5), padding='same', activation='relu', kernel_initializer='uniform')(max_pool1)
conv2 = Conv2D(32, kernel_size=(5,5), padding='same', activation='relu', kernel_initializer='uniform')(conv2)
conv2 = Conv2D(32, kernel_size=(5,5), padding='same', activation='relu', kernel_initializer='uniform')(conv2)
bn2 = BatchNormalization(scale=True)(conv2)

    # First Residual
res_conv1 = Conv2D(32, kernel_size=(3,3), padding='same', kernel_initializer='uniform')(max_pool1)
res_bn1 = BatchNormalization(scale=True)(res_conv1)

    # First Add
add1 = Add()([res_bn1, bn2])

    # First Acvtivation & MaxPooling
act1 = Activation('relu')(add1)
max_pool2 = MaxPooling2D(pool_size=(2,2))(act1)

    # Second
conv3 = Conv2D(64, kernel_size=(5,5), padding='same', activation='relu', kernel_initializer='uniform')(max_pool2)
conv3 = Conv2D(64, kernel_size=(5,5), padding='same', activation='relu', kernel_initializer='uniform')(conv3)
conv3 = Conv2D(64, kernel_size=(5,5), padding='same', activation='relu', kernel_initializer='uniform')(conv3)
bn3 = BatchNormalization(scale=True)(conv3)

    # Second Residual
res_conv2 = Conv2D(64, kernel_size=(3,3), padding='same', kernel_initializer='uniform')(max_pool2)
res_bn2 = BatchNormalization(scale=True)(res_conv2)

    # Second Add
add2 = Add()([res_bn2, bn3])

    # Second Acvtivation & MaxPooling
act2 = Activation('relu')(add2)
max_pool3 = MaxPooling2D(pool_size=(2,2))(act2)

    # Flattern the data
flatten = Flatten()(max_pool3)

    # Fully Connected Layer
dense1 = Dense(128, activation='relu')(flatten)
do = Dropout(0.25)(dense1)

output = Dense(9, activation='softmax')(do)
    #print(output)
    # bind all
model = Model(inputs=model_original.input, outputs=output)

base_learning = 0.0001
mod_HeadPose .compile(
    optimizer=Adam(learning_rate=base_learning),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

mod_HeadPose .load_weights("C:/Users/CESAR/OneDrive/Documentos/CI/Face_Analysis/FOrien/weights_CNN_HeadPose.h5")

mod_HeadPose .save("HeadPose.h5")