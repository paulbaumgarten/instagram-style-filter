import cv2
import numpy as np
import os, sys, time, json
from PIL import Image


def take_photo( camera_device_id=0, width=640, height=480, flip=False ):
    cap = cv2.VideoCapture(camera_device_id)
    cap.set(3, camera_width)
    cap.set(4, camera_height)
    ret, cv2_img = cap.read()
    assert ret, "Error reading from capture device "+str(camera_device_id)
    if flip:
        cv2_img = cv2.flip(cv2_img, -1)
    pil_img = Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))
    return pil_img

def detect_faces( image, cascade_file="haarcascade_frontalface_default.xml", testing_mode=False):
        # convert PIL image back to CV2 numpy array
        img = cv2.cvtColor(np.array(background), cv2.COLOR_RGB2BGR)
    if not os.path.exists(cascade_file):
        print(f"Cascade file not found '{cascade_file}'")
        print("You may need to download it from https://github.com/opencv/opencv/tree/master/data/haarcascades")
        print("Aborting....")
        exit(1)
    min_detect_width = 200
    min_detect_height = 200
    classifier = cv2.CascadeClassifier(cascade_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = classifiers["face"].detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,     
        minSize=(min_detect_width, min_detect_height)
    )
    if testing_mode:
        for (face_x, face_y, face_w, face_h) in faces:
            # Draw a rectangle around the face
            cv2.rectangle(img, (face_x,face_y), (face_x+face_w,face_y+face_h), (255,0,0), 2)
    return faces

def detect_left_eye( image, cascade_file="haarcascade_mcs_lefteye.xml" ):
    pass

def detect_right_eye( image, cascade_file="patterns/haarcascade_mcs_righteye.xml" ):
    pass

def detect_nose( image, cascade_file="patterns/haarcascade_mcs_nose.xml" ):
    pass


## Setup
filter_file             = "./filters/snapchat-filters-png-4071.png"
filter_im = Image.open(filter_file)

## Prompt for ready
input("Are you ready for me to take your photo?")

photo = take_photo(0)
photo.show()


        # Isolate the face
        face_image_gray = gray[face_y:face_y+face_h, face_x:face_x+face_w]
        face_image_color = img[face_y:face_y+face_h, face_x:face_x+face_w]

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
                    #cv2.circle(face_image_color, feature_centre, 5, (0,255,0), -1)
                    #cv2.putText(face_image_color, key, (fx+5,fy-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

        # Execute the callback, alerting we found a face!
        #if callback is not None:
        #    loop = callback({"x":x,"y":y,"w":w,"h":h}, img)

        # convert CV2 numpy array to PIL image
        background = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        filter_im = filter_im.resize((face_w,int(face_h*1.5)))
        filter_coordinates = (face_x, face_y, face_x+face_w, face_y+int(face_h*1.5))
        background.paste(filter_im, filter_coordinates, filter_im)
        # convert PIL image back to CV2 numpy array
        img = cv2.cvtColor(np.array(background), cv2.COLOR_RGB2BGR)
    cv2.imshow('video',img)

    # Check for exit key press
    k = cv2.waitKey(30) & 0xff
    if k == 27: # press 'ESC' to quit
        loop = False

cap.release()
cv2.destroyAllWindows()
