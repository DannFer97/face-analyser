#Librerias
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D



    #from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
model = MobileNetV2(input_shape=(64,64,3),include_top=False,weights="imagenet")

 for layer in model.layers[:-6]:
    layer.trainable=False

x =model
x = Conv2D(64, kernel_size=(3, 3), activation='relu')(x)

    # Añade una nueva capa de max-pooling con un tamaño de pool de (2, 2)
x = MaxPooling2D(pool_size=(2, 2))(x)

    # Añade una nueva capa de conv2D con 64 filtros y un tamaño de kernel de (3,3)
x = Conv2D(64, kernel_size=(3, 3), activation='relu')(x)

    # Añade una nueva capa de max-pooling con un tamaño de pool de (2, 2)
 x = MaxPooling2D(pool_size=(2, 2))(x)

    # Añade una nueva capa de aplanamiento
x = Flatten()(x)

    # Añade una nueva capa densa con 512 unidades y una función de activación ReLU
x = Dense(512, activation='relu')(x)

    # Añade una nueva capa densa de salida con 9 unidades y una función de activación softmax
output = Dense(9, activation='softmax')(x)
modelo = Model(inputs=model.input, outputs=output)

base_learning = 0.001
modelO.compile(
    optimizer=Adam(learning_rate=base_learning),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

modelo.load_weights("C:\Users\CESAR\OneDrive\Documentos\CI\Face_Analysis\FOrien\weights_CNN_HeadPosev8.h5")
modelo.save("HeadPose.h5")