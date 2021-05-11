import cv2
import numpy as np

video = cv2.VideoCapture(0)
classificadorFace = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
classificadorOlhos = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

while True: 
  conectado, frame = video.read()
  frameCinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  facesDetectadas = classificadorFace.detectMultiScale(frameCinza, minSize=(50,50))
  for (x,y,l,a) in facesDetectadas:
    cv2.rectangle(frame, (x,y), (x + l, y + a), (0,255,0),2)
    regiao = frame[y:y + a, x:x + l]
    frame_gaussian = cv2.GaussianBlur(regiao,(101,101),0)
    frame[y:y + a, x:x + l] = frame_gaussian
  cv2.imshow('Video', frame)

  if cv2.waitKey(1) == ord('q'):
    break

video.release()
cv2.destroyAllWindows();
