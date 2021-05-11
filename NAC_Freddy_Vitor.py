from pynput import mouse
import cv2
import numpy as np

num_down = 2
num_bilateral = 7
video = cv2.VideoCapture(0)
count = 0;
faceClassifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
def cartoon_filter(img_rgb):
    for _ in range(num_down):
        img_rgb = cv2.pyrDown(img_rgb)
    for _ in range(num_bilateral):
        img_rgb = cv2.bilateralFilter(img_rgb, d=9, sigmaColor=9, sigmaSpace=7)
    for _ in range(num_down):
        img_rgb = cv2.pyrUp(img_rgb)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    img_blur = cv2.medianBlur(img_gray, 7)
    img_edge = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=9, C=2)
    img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)
    img_cartoon = cv2.bitwise_and(img_rgb, img_edge)
    return img_cartoon
def blur_filter(img_rgb):
  img_blur = cv2.GaussianBlur(img_rgb,(101,101),0)
  return img_blur
def estatua_filter(img_rgb):
  img_estatua = cv2.stylization(img_rgb, sigma_s=30, sigma_r=0.57)
  return img_estatua
def on_click(event, x, y, flags, param):   
    global count
    if event == cv2.EVENT_RBUTTONDOWN:
        count = 0
        print(count)
        pass
    elif event == cv2.EVENT_LBUTTONDOWN:
      print(count)
      if count == 3 :
        count = 1
      else:
        count += 1
        pass
while True: 
  conectado, frame = video.read() 
  conectado, videoOriginal = video.read()
  frameCinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  faces_return = faceClassifier.detectMultiScale(frameCinza, minSize=(100,100))
  if(count == 0):
    for (x,y,l,a) in faces_return:
      
      frame[y:y + a, x:x + l]
  elif count == 1 :
    for (x,y,l,a) in faces_return:
      blur = blur_filter(frame)
      face_local = blur[y:y + a, x:x + l]
      frame[y:y + a, x:x + l] = face_local
  elif count == 2 :
    for (x,y,l,a) in faces_return:
      cartoon = cartoon_filter(frame)
      face_local = cartoon[y:y + a, x:x + l]
      frame[y:y + a, x:x + l] = face_local
  elif count == 3 :
    for (x,y,l,a) in faces_return:
      estatua = estatua_filter(frame)
      face_local = estatua[y:y + a, x:x + l]
      frame[y:y + a, x:x + l] = face_local
  if cv2.waitKey(1) == ord('q'):
    break
  cv2.setMouseCallback('VideoFiltro', on_click)
  cv2.imshow('VideoFiltro', frame)
  cv2.imshow('VideoOriginal', videoOriginal)
video.release()
cv2.destroyAllWindows();