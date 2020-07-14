import  cv2
import  os
import numpy as np
import faceRecognition as fr
#print('hello tester')

test_img = cv2.imread('testimages/cr7_test.jpg')
face_detected, gray_img = fr.faceDetection(test_img)
print('faces detected:',face_detected)

#for (x, y, w, h) in face_detected:
#    cv2.rectangle(test_img, (x, y), (x+w, y+h), (255, 0, 0), thickness=2)
#resized_img = cv2.resize(test_img, (600, 400))
#cv2.imshow('Face Detected',resized_img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

faces, faceID = fr.labels_for_training_data('trainingimages')
face_recognizer = fr.train_classifier(faces, faceID)
face_recognizer.save('trainingData.yml')

name = {0:'Akshay Kumar', 1:'Cristiano Ronaldo'}

for face in face_detected:
    (x,y,w,h) = face
    roi_gray = gray_img[y:y+w, x:x+h]
    label, confidence = face_recognizer.predict(roi_gray)
    print('Confidence : ',confidence)
    print('Label : ', label)
    fr.draw_rect(test_img, face)
    predicted_name = name[label]
    fr.put_text(test_img, predicted_name,x,y)

resized_img = cv2.resize(test_img, (800, 600))
cv2.imshow('Face Detected',resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


