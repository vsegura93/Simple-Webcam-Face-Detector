# Import libraries
import cv2
import sys
import logging as log
import datetime as dt
from time import sleep

# Load trained models of face and eyes
faceClassifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eyesClassifier = cv2.CascadeClassifier('haarcascade_eye.xml')

# Load video capture from webcam
webcamVideo = cv2.VideoCapture(0)
last = 0

while True:

    # Check webcam device
    if not webcamVideo.isOpened():
        print('Unable to load camera. Check webcam connection.')
        sleep(5)
        pass

    # Capture frame-by-frame
    ret, frame = webcamVideo.read()

    # Transform frame to gray scale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Load face detector using multiscale approach
    faces = faceClassifier.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    if last != len(faces):
        last = len(faces)
        log.info("faces: "+str(len(faces))+" at "+str(dt.datetime.now()))

    for (x,y,w,h) in faces:	
	cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
	roi_gray = gray[y:y+h, x:x+w]
	roi_color = frame[y:y+h, x:x+w]
	eyes = eyesClassifier.detectMultiScale(roi_gray)

	# Draw a rectangle around the eyes
	for (ex,ey,ew,eh) in eyes:
	    cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Display the resulting frame
    cv2.imshow('Face detector ( press "q" to exit )', frame)

# When everything is done, release the capture and destroy window
webcamVideo.release()
cv2.destroyAllWindows()
