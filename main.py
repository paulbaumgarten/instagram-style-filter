import cv2
import numpy as np
import os, sys, time, json, math
from PIL import Image
from PIL import ImageFilter

def hex_to_rgb(value):
    """ Converts `#ffffff` to (255,255,255) """
    value = value.lstrip('#')
    r = int(value[:2], 16)
    g = int(value[2:4], 16)
    b = int(value[4:], 16)
    return (r,g,b)

class camera:
    def __init__(self, camera_device=0, resolution=(1920,1080), flip=False):
        width, height = resolution
        self.camera_device = camera_device
        self.width = width
        self.height = height
        self.flip = flip

    def get_photo(self ):
        cap = cv2.VideoCapture(self.camera_device)
        cap.set(3, self.width)
        cap.set(4, self.height)
        ret, cv2image = cap.read()
        assert ret, "Error reading from capture device "+str(camera_device)
        if self.flip:
            cv2image = cv2.flip(cv2image, -1)
        pilimage = Image.fromarray(cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGB))
        return pilimage

    def save_photo(self, filename):
        pilimage = self.get_photo()
        pilimage.save(filename)
        return pilimage
    
    @staticmethod
    def list_camera_modes():
        pass

def image_remove_background( source_filename, target_filename, background_color="#ffffff", sensitivity=0.1):
    """ sensitivity is 0..1 """
    sensitivity_per_byte = int(256*sensitivity)
    img = Image.open(source_filename)  
    img = img.convert("RGBA")  
    remove_color = hex_to_rgb(background_color)
    pixdata = img.load()  
    for y in range(img.size[1]):  
        for x in range(img.size[0]):  
                r, g, b, a = img.getpixel((x, y))  
                if abs(remove_color[0]-r)<sensitivity_per_byte and abs(remove_color[1]-g)<sensitivity_per_byte and abs(remove_color[2]-b)<sensitivity_per_byte:
                    pixdata[x, y] = (0, 0, 0, 0)  # make it transparent
    img2 = img.filter(ImageFilter.GaussianBlur(radius=1))  
    if target_filename[-4:].lower() in (".jpg", ".jpe"):
        target_filename = target_filename[:-4] + ".png"
    img2.save(target_filename, "PNG")  



## Prompt for ready
input("Are you ready for me to take your photo?")

cam = camera(0, (1920,1080))
photo = cam.save_photo("bleh.jpg")
photo.show()

image_remove_background("thanos.jpg", "new.png", background_color="#00cb15")


"""
input_img = Image.open("thanos.jpg")
output_img = Image.new("RGBA", input_img.size)

for y in range(input_img.size[1]):
	for x in range(input_img.size[0]):
		p = input_img.getpixel((x, y))
		d = math.sqrt(math.pow(p[0], 2) + math.pow((p[1] - 255), 2) + math.pow(p[2], 2))
	
		if d > 128:
			d = 255
	
		output_img.putpixel((x, y), (p[0], min(p[2], p[1]), p[2], int(d)))

output_img.save("thanos.png", "PNG")

"""

"""
def detect_faces( image, cascade_file="haarcascade_frontalface_default.xml", testing_mode=False):
    # convert PIL image back to CV2 numpy array
    cv2image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    if not os.path.exists(cascade_file):
        print(f"Cascade file not found '{cascade_file}'")
        print("You may need to download it from https://github.com/opencv/opencv/tree/master/data/haarcascades")
        print("Aborting....")
        exit(1)
    min_detect_width = 200
    min_detect_height = 200
    classifier = cv2.CascadeClassifier(cascade_file)
    gray = cv2.cvtColor(cv2image, cv2.COLOR_BGR2GRAY)
    faces = classifiers["face"].detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,     
        minSize=(min_detect_width, min_detect_height)
    )
    if testing_mode:
        for (face_x, face_y, face_w, face_h) in faces:
            # Draw a rectangle around the face
            cv2.rectangle(cv2image, (face_x,face_y), (face_x+face_w,face_y+face_h), (255,0,0), 2)
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
"""


"""
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
"""