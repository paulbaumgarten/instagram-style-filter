import cv2
import numpy as np
import os, sys, time, json
from PIL import Image

## Setup
images_folder="."
patterns = {}
patterns["face"]        = "patterns/haarcascade_frontalface_default.xml"
#patterns["leftear"]     = "patterns/haarcascade_mcs_leftear.xml"
patterns["lefteye"]     = "patterns/haarcascade_mcs_lefteye.xml"
#patterns["rightear"]    = "patterns/haarcascade_mcs_rightear.xml"
patterns["righteye"]    = "patterns/haarcascade_mcs_righteye.xml"
#patterns["mouth"]       = "patterns/haarcascade_mcs_mouth.xml"
patterns["nose"]        = "patterns/haarcascade_mcs_nose.xml"
#patterns["eyes"]        = "patterns/haarcascade_mcs_eyepair_big.xml"

flip = False
camera_device_id = 0
camera_width = 640
camera_height = 480
min_detect_width = 200
min_detect_height = 200

## Get camera
# cv2.namedWindow("preview") # Mac only
cap = cv2.VideoCapture(camera_device_id)
cap.set(3, camera_width)
cap.set(4, camera_height)

## Initialise cascades
classifiers = {}
for key,val in patterns.items():
    classifiers[key] = cv2.CascadeClassifier(val)

## Start the loop
loop = True
while loop:

    # Read image from the camera
    ret, img = cap.read()
    assert ret, "Error reading from capture device "+str(camera_device_id)
    if flip:
        img = cv2.flip(img, -1)

    # Convert image to grey scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect any faces in the image? Put in an array
    faces = classifiers["face"].detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,     
        minSize=(min_detect_width, min_detect_height)
    )

    # For every face we found
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)

        # Isolate the face
        face_image_gray = gray[y:y+h, x:x+w]
        face_image_color = img[y:y+h, x:x+w]

        # Search for face features
        for key,feature in classifiers.items():
            if key != "face":
                feature_coords = feature.detectMultiScale(
                    face_image_gray,
                    scaleFactor=1.2,
                    minNeighbors=10,     
                    minSize=(min_detect_width//6, min_detect_height//6),
                    maxSize=(min_detect_width//3, min_detect_height//3)
                )
                for (fx, fy, fw, fh) in feature_coords:
                    # cv2.rectangle(face_image_color, (fx,fy), (fx+fw,fy+fh), (0,255,0), 1)
                    feature_centre = fx + fw//2, fy + fh//2
                    cv2.circle(face_image_color, feature_centre, 5, (0,255,0), -1)
                    cv2.putText(face_image_color, key, (fx+5,fy-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

        # Execute the callback, alerting we found a face!
        #if callback is not None:
        #    loop = callback({"x":x,"y":y,"w":w,"h":h}, img)

    # Show the face on screen
    cv2.imshow('video',img)

    # Check for exit key press
    k = cv2.waitKey(30) & 0xff
    if k == 27: # press 'ESC' to quit
        loop = False

cap.release()
cv2.destroyAllWindows()
