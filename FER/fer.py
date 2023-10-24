#Librerias
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D


def generar_modelo():
    from tensorflow.keras.applications import ResNet50
    model = ResNet50(input_shape=(48,48,3),include_top=False,weights="imagenet")

    for layer in model.layers[:-4]:
        layer.trainable=False
    
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))
    
    return model
    
def color_expression(indice):
    if indice==0:
        R,G,B=208,60,28
    elif indice==1:
        R,G,B=25,186,42
    elif indice==2:
        R,G,B=4,4,4
    elif indice==3:
        R,G,B=243,232,4
    elif indice==4:
        R,G,B=255,255,255
    elif indice==5:
        R,G,B=9,65,158
    elif indice==6:
        R,G,B=181,4,243
    else:
        R,G,B=255,255,255
    
    return (R,G,B)

def porcentaje(expression):
    expression = expression*100
    expression = expression.astype(int)
    return expression


def predecir_expression(model,imagen,x,y,w,h):
    # bbox_array=np.zeros([480,640,4],dtype=np.uint8)
    #Diccionario de emociones
    #diccionario_emocion = {0: "Enojo", 1: "Disgusto", 2: "Miedo", 3: "Feliz", 4: "Neutral", 5: "Triste", 6: "Sorpresa"}
    
    imagen = np.stack([imagen]*3, 2)
    imagen_prediccion = np.expand_dims(cv2.resize(imagen, (128, 128)), 0)
    imagen_prediccion = imagen_prediccion.astype(float)
    imagen_prediccion /= 255
    prediccion = model.predict(imagen_prediccion)
    expressions = porcentaje(prediccion[0])
    altura = y + (np.divide(h, 4.0))
    altura = altura.astype(int)
    
    return altura, expressions

def prediction_print_expressions(modelo_fer, diccionario_expresiones, frame, x, w, y, h):
    face_cropped = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    face_cropped = face_cropped[y:y + h, x:x + w]
    try:
        altura, expreciones = predecir_expression(modelo_fer, face_cropped, x, y, w, h)
    

        # Imprimir expresiones faciales        
        for i in range(7):
            cv2.putText(frame, diccionario_expresiones[i]+'='+str(expreciones[i]),
                            (x+w, altura+(i*15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            color_expression(i), 1, cv2.LINE_AA)
    except:
        pass
    return
 
