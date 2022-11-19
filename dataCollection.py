import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import os
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset =20
imgSize = 400
counter = 0

folder = "images/Four"

while True:
    sucess, img = cap.read()
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

        
        else:
            k = imgSize/w
            hCalc = math.ceil(h*k)

            hGap = math.ceil((imgSize-hCalc)/2)

            imgResize = cv2.resize(imgCrop,(imgSize,hCalc))
            imageResizeShape = imgResize.shape
            imageWhite[hGap:hCalc+hGap,:] = imgResize


        cv2.imshow("ImageSquared",imageWhite)

    cv2.imshow("VideoLive", img)
    key = cv2.waitKey(1)

    if key == ord('s'):
        counter += 1
        cv2.imwrite(f"{folder}/Image_{time.time()}.jpg",imageWhite)
        print(counter)
