'''''
Haar-Cascade vehicle detect
'''''

import pandas as pd
import numpy as np
import cv2
from PIL import Image
import requests

img = Image.open(requests.get('https://ichef.bbci.co.uk/news/976/cpsprodpb/1F7F/production/_123236080_nationalhighwaysm27.png', stream=True).raw) 
img = img.resize((450,250))
img_array = np.array(img)
#img.show()

#image manipulation
to_gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
img=Image.fromarray(to_gray)
#img.show()

#Gaussian Blur
gauss_blur = cv2.GaussianBlur(to_gray,(5,5),0)
img=Image.fromarray(gauss_blur)
#img.show()

#image dilation
dil=cv2.dilate(gauss_blur, np.ones((3,3)))
img=Image.fromarray(dil)
#img.show()

#kernel - morphology transformation
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
morph_close=cv2.morphologyEx(dil, cv2.MORPH_CLOSE, kernel)
img=Image.fromarray(morph_close)
#img.show()


#car-cascade detection
car_cascade_src ='/Users/danie/Desktop/Projects/Portfolio/VehicleDetection/detection/vehicleDetection/Files/cars.xml'
#car_cascade_src = '/content/drive/MyDrive/Projects/VehicleDetection/Collab+Code/Collab Code/cars.xml'
car_cascade = cv2.CascadeClassifier(car_cascade_src)
cars = car_cascade.detectMultiScale(morph_close, 1.1, 1)

contour = 0
for(x,y,w,h) in cars:
    cv2.rectangle(img_array,(x,y),(x+w, y+h),(255,0,0),2)
    contour += 1
print(contour, ' cars found')
img=Image.fromarray(img_array)    
img.show()

#Haar-Cascade video
haar_source = '/Users/danie/Desktop/Projects/Portfolio/VehicleDetection/detection/vehicleDetection/Files/cars.xml'
video_source = '/Users/danie/Desktop/Projects/Portfolio/VehicleDetection/detection/vehicleDetection/Files/cars.mp4'
capture = cv2.VideoCapture(video_source)
car_cascade = cv2.CascadeClassifier(haar_source)
video = cv2.VideoWriter('/Users/danie/Desktop/Projects/Portfolio/VehicleDetection/detection/vehicleDetection/Files/result115.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 25, (640,360)) 

while True:
    ret, img = capture.read()
   
    if (type(img) == type(None)):
        break
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cars = car_cascade.detectMultiScale(gray, 1.15, 2) #1.15 seems to work without misclassification (1.1 = nope)

    for (x,y,w,h) in cars:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)

    video.write(img) 
video.release()
print('video ok')