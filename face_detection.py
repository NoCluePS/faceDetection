import cv2
import urllib.request
import os
import validators
from random import randrange

# Load data into py file
trained_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
real_time = input('Real-time? (y/n) ') == 'y';

if (not real_time):
    url = input('Image URl: ');
    save_name = 'head.jpg';

    img = '';

    while (not validators.url(url)):
        print('Invalid URL')
        url = input('Image URL: ')

    urllib.request.urlretrieve(url, save_name);
    img = cv2.imread(save_name);

    grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_coordinates = trained_data.detectMultiScale(grayscaled_img);
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(img, (x, y), (x+w, y+h), (randrange(0, 255), randrange(0, 255), randrange(0, 255)), 5)
        cv2.putText(img, 'Face', (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Face detection', img);
    cv2.waitKey();

    os.remove('head.jpg');
else:
    webcam = cv2.VideoCapture(0);

    while True:
        successful_frame_read, frame = webcam.read()
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY);
            
        face_coordinates = trained_data.detectMultiScale(grayscaled_frame);
        for (x, y, w, h) in face_coordinates:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 5)
            cv2.putText(frame, 'Face', (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow('Face detection', frame);
        # Gets ascii value of the key
        key = cv2.waitKey(1);

        if key == 82 or key == 113:
            break
