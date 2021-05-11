import cv2
import numpy as np
num_down = 2
num_bilateral = 7
video = cv2.VideoCapture(0)

classificadorFace = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

while True: 
  conectado, frame = video.read()
  #print(frame)
  
  frameCinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  facesDetectadas = classificadorFace.detectMultiScale(frameCinza, minSize=(100,100))

  edges = cv2.Canny(frame,75,150)
  
  for (x,y,l,a) in facesDetectadas:
    cv2.rectangle(frame, (x,y), (x + l, y + a), (0,255,0),2)
    regiao = frame[y:y + a, x:x + l]
    dst = cv2.stylization(regiao, sigma_s=30, sigma_r=0.57)
    frame[y:y + a, x:x + l] = dst


  cv2.imshow('Video', frame)
  if cv2.waitKey(1) == ord('q'):
    break

video.release()
cv2.destroyAllWindows();