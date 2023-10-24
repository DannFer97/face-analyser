import cv2

def face_position(frame, detectorSSD):
  # inverted image for face detection
  baseImage = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
  baseImage = cv2.resize(baseImage,(320,240))
  imageBlob = cv2.dnn.blobFromImage(image = baseImage)

  detectorSSD.setInput(imageBlob)
  faces = SSD_2_rectangles(frame, detectorSSD.forward(), 0.70)

  return faces
  
def SSD_2_rectangles(frame, detections, th):
    faces = []
    resizeW, resizeH = frame.shape[1], frame.shape[0]
    for i in range(detections[0][0].shape[0]):
      if(detections[0][0][i][2]>th):
        x1 = int(resizeW*detections[0][0][i][3])
        y1 = int(resizeH*detections[0][0][i][4])
        r1 = int(resizeW*detections[0][0][i][5])
        b1 = int(resizeH*detections[0][0][i][6])
        
        faces.append([x1,y1,r1-x1,b1-y1])
    return faces