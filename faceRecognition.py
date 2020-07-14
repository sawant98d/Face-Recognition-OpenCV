import cv2
import numpy as np
import os

def faceDetection(test_img):
    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)  # converting img to gray image
    face_haar_cascade = cv2.CascadeClassifier('HaarCascade/haarcascade_frontalface_default.xml')   # using face haarcascade frontalface classifier
    faces = face_haar_cascade.detectMultiScale(gray_img, scaleFactor=1.32, minNeighbors=5)

    return faces, gray_img

def labels_for_training_data(directory):
    faces = []
    faceID = []

    for path, subdirnames, filenames in os.walk(directory):
        for filename in filenames:
            if filename.startswith('.'):
                print('Skipping system files')
                continue
            id = os.path.basename(path)
            img_path = os.path.join(path, filename)
            print('img_path : ', img_path)
            print('ID : ', id)
            test_img = cv2.imread(img_path)
            if test_img is None:
                print('Image not loaded properly.')
                continue

            faces_rect, gray_img = faceDetection(test_img)
            if len(faces_rect) != 1:
                continue # Since we are assuming only single person images are being fed to classifier

            (x, y, w, h) = faces_rect[0]   # single detected face values
            roi_gray = gray_img[y:y+w, x:x+h]  # cropping the only gray image of detected face
            faceID.append(int(id))
            faces.append(roi_gray)
    return faces, faceID


def train_classifier(faces, faceID):
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(faces, np.array(faceID))
    return face_recognizer

def draw_rect(test_img, face):
    (x,y,w,h) = face
    cv2.rectangle(test_img, (x,y), (x+w,y+h), (255,0,0), thickness=5)

def put_text(test_img, text, x, y):
    cv2.putText(test_img, text, (x,y), cv2.FONT_HERSHEY_DUPLEX, 1, (255,0,0), 1)










