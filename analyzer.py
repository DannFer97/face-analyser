#------- Librerias para GUI ------------
import tkinter as tk
import cv2
from PIL import Image, ImageTk

#-------- Librerias para IA ------------
import numpy as np
from tensorflow.keras.models import load_model

#------- Subprogramas -----------------
from Face_Detector import ssd_detector
from threading import Thread
from FER import fer
from FOrien import faceOrien

#------- Carga de los Modelos ----------
print("Cargando modelos de reconocimiento ...")

# Rutas de los modelos
directorio_modelos = "Modelos"
rutah5fer = directorio_modelos + "/fer_model.h5"             # Ruta del modelo entrenado
dirSSD = directorio_modelos + "/ssd/"
rutaHeadPose= directorio_modelos + "/HeadPose.h5"

print("Cargando modelo de expresiones...")
modelo_fer = load_model(rutah5fer)                         # Creación de modelo emociones
print("Modelo de expresiones cargado.")

print("Cargando modelo de orientación...")
modelo_headP = load_model(rutaHeadPose)                         # Creación de modelo emociones
print("Modelo de orientaciones.")

print("Inicializando modelo de reconocimiento de rostros...")
detectorSSD = cv2.dnn.readNetFromCaffe(dirSSD+"deploy.json", dirSSD+"res10_300x300_ssd_iter_140000.caffemodel")
print("Modelo de reconocimiento inicializado.")

#--------- Constantes de ejecución --------------
#diccionario_expresiones = {0: "Enojo", 1: "Neutral", 2: "Disgusto", 3: "Miedo", 4: "Feliz", 5: "Triste", 6: "Sorpresa"}
size = 1

diccionario_orientaciones = {0: "Derecha Arriba" , 1: " Arriba", 2: "Izquierda Arriba", 3: "Derecha",
 4: "Centro", 5: "Izquierda", 6: "Derecha Abajo",7: "Abajo", 8: "Izquierda Abajo"}
def onClossing():
    root.quit()
    cap.release()
    print("Camera Disconnected")
    root.destroy()

def callback():
    frame_count = 0
    ret, frame = cap.read()
    if not ret:
        onClossing()
    
    frame = cv2.flip(frame, 1)

    faces = ssd_detector.face_position(frame, detectorSSD)

    if faces: # Para reconocimiento de un rostro a la vez por cuadro
        face_i = faces[0]
        (x, y, w, h) = [v * size for v in face_i]
        #face_cropped = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #face_cropped = face_cropped[y-40:y+h+25, x-25:x+w+35]


        #th_face_box = Thread(target=cv2.rectangle, args = (frame, (x-40, y-40), (x+w+40, y+h+70), (50, 255, 50), 3))
        th_face_box = Thread(target=cv2.rectangle,
                             args=(frame, (x-25 , y-40 ), (x + w +35, y + h+25 ), (50, 255, 50), 3))
        #th_fer_print = Thread(target=fer.prediction_print_expressions, args=(modelo_fer, diccionario_expresiones, frame, x, w, y, h))
        th_headP_print = Thread(target=faceOrien.prediction_print_orientation,
                              args=(modelo_headP, diccionario_orientaciones, frame, x, w, y,h,frame_count))
        frame_count += 1
        th_headP_print.start()
        th_face_box.start() # Colocar cuadro de rostro
        #th_fer_print.start() # Imprimir expresiones faciales

        
        th_face_box.join()
        #th_fer_print.join()
        th_headP_print.join()

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    img = Image.fromarray(img)
    img = ImageTk.PhotoImage(img)
    label.configure(image = img)
    label.image = img
    root.after(32, callback)



cap = cv2.VideoCapture(0)

root = tk.Tk()
root.protocol("WM_DELETE_WINDOW", onClossing)

root.title("Video")

label = tk.Label(root)
label.pack()

root.after(32, callback)
root.mainloop()
