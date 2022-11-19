import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import os
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5","Model/labels.txt")

offset =20
imgSize = 400
counter = 0

labels = ["A","B","C","One","Two","Three","Four","Five","Six","Seven","Eight","Nine"]


folder = "images/Four"

while True:
    sucess, img = cap.read()
    imgOutput = img.copy()

    hands , img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x,y,w,h = hand["bbox"]

        imageWhite = np.ones((imgSize,imgSize,3),np.uint8)*255


        

        imgCrop = img[y-offset:y+h+offset, x-offset:x+w+offset]
        cv2.imshow("ImageCropped", imgCrop)



        aspectRatio = h/w

        if aspectRatio > 1:
            k = imgSize/h
            wCalc = math.ceil(w*k)

            wGap = math.ceil((imgSize-wCalc)/2)

            imgResize = cv2.resize(imgCrop,(wCalc,imgSize))
            imageResizeShape = imgResize.shape
            imageWhite[:,wGap:wCalc+wGap] = imgResize
            prediction,index = classifier.getPrediction(img)
            print(labels[index])
        
        else:
            k = imgSize/w
            hCalc = math.ceil(h*k)

            hGap = math.ceil((imgSize-hCalc)/2)

            imgResize = cv2.resize(imgCrop,(imgSize,hCalc))
            imageResizeShape = imgResize.shape
            imageWhite[hGap:hCalc+hGap,:] = imgResize
            prediction,index = classifier.getPrediction(img)
        cv2.rectangle(imgOutput, (x - offset, y - offset-50),
                      (x - offset+90, y - offset-50+50), (255, 0, 255), cv2.FILLED)
        cv2.putText(imgOutput, labels[index], (x, y -26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
        cv2.rectangle(imgOutput, (x-offset, y-offset),
                      (x + w+offset, y + h+offset), (255, 0, 255), 4)



        cv2.imshow("ImageSquared",imageWhite)

    cv2.imshow("VideoLive", img)
    