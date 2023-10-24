#Librerias
import numpy as np
import cv2
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from PIL import Image


def color_expression(indice):
    if indice==10:
        R,G,B=208,60,28
    else:
        R,G,B=0,0,0
    
    return (R,G,B)

def porcentaje(expression):
    expression = expression*100
    expression = expression.astype(int)
    return expression


def predecir_orientacion(model,imagen,x,y,w,h,frame_count):

    #imagen = np.stack([imagen]*3, 2)
    imagen= cv2.flip(imagen,1)
    #face_cropped = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    face_cropped = imagen[y-40:y+h+55, x-25:x+w+300]
   # face_cropped= imagen [(x-25 , y-40 ), (x + w +35, y + h+25 )]
    #face_cropped = face_cropped[y-40:y+h+55, x-25:x+w+55]

    cv2.imwrite('./save/frame_{}.jpg'.format(frame_count), cv2.resize(face_cropped, (128, 128)))

    imagen_prediccion = np.expand_dims(cv2.resize(face_cropped, (128, 128)), 0)
    #imagen_prediccion = imagen_prediccion.astype(float)
    imagen_prediccion = imagen_prediccion/255

    #y_pred = np.argmax(new_model_HeadPose.predict(test_images), axis=-1)
    prediccion = model.predict(imagen_prediccion)
    cv2.imshow("Predictions", prediccion)
    orientations = prediccion[0]
    print(orientations)
    altura = y + (np.divide(h, 4.0))
    altura = altura.astype(int)
    return altura, orientations

def prediction_print_orientation(modelo_headP, diccionario_orientaciones, frame, x, w, y, h,frame_count):
    #face_cropped = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    max_val = 0
    max_key = None
    #face_cropped = face_cropped[y:y + h, x:x + w]
    try:
        altura, orient = predecir_orientacion(modelo_headP, frame, x, y, w, h,frame_count)
    

        # Imprimir orientaciones faciales
        for i in range(9):
            if orient[i] > max_val:
                max_val = orient[i]
                max_key = i
        color = (255, 0, 0)
        cv2.putText(frame, diccionario_orientaciones[max_key] + '=' + str(max_val),
                    (x + w, altura),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    color, 1, cv2.LINE_AA)

    except:
        pass
    return
 
