import cv2
import numpy as np
import os

def checkDataset(dir="dataset/"):
    if not os.path.exists(dir) and len(os.listdir(dir)) != 0:
        return True
    return False

def organizeDataset(dir="dataset/"):
    imagePath = [os.path.join(dir, f) for f in os.listdir(dir)]
    faces = []
    ids = np.array([], dtype=np.int32)
    for i in imagePath:
        img = cv2.cvtColor(cv2.imread(i), cv2.COLOR_BGR2GRAY)
        id = int(i.split()[-1].split(". ")[0].split("-")[1])
        face = faceCascade.detectMultiScale(img)
        for (x, y, w, h) in face:
            faces.append(img[y:y+h, x:x+w])
            ids = np.append(ids, id)
    return faces, ids
if not checkDataset():
    print("Dataset not found")
else:
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    faces, ids = organizeDataset()

    print("Training faces...")
    recognizer.train(faces, ids)
    print("Training completed.")
    recognizer.write("face-model.yml")
    print("Model saved as face-model.yml")