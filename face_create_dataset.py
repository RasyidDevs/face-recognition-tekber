import cv2
import os

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)
dataset_path = "dataset/"
person_name = "Rasyid"
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)
if not os.path.exists(dataset_path + person_name):
    os.makedirs(dataset_path + person_name)
dict_name = {"Rasyid": [(0, 0, 255)], "Faruq": [(0, 255, 0)], "Akbar": [(255,0,0)]}
count = len(os.listdir(dataset_path + person_name))
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        count += 1
        cv2.rectangle(frame, (x,y), (x+w, y+h), dict_name.get(person_name)[0], 2)
        cv2.putText(frame, str(count), (x+5, y-5), cv2.FONT_HERSHEY_COMPLEX, 1, dict_name.get(person_name)[0], 1)
        cv2.putText(frame, person_name, (x+5, y+h-5), cv2.FONT_HERSHEY_COMPLEX, 1, dict_name.get(person_name)[0], 1)
        cv2.imwrite(dataset_path + "/" + person_name + "/Person-" + person_name + "-" + str(count) + ".jpg", gray[y:y+h, x:x+w])
    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) == ord("q"):
        break
    elif count == 350:
        break
cap.release()
cv2.destroyAllWindows()