import cv2
import os
import numpy as np
from insightface.app import FaceAnalysis

dict_name = {
    "Rasyid": (0, 0, 255),
    "Faruq": (0, 255, 0),
    "Akbar": (255, 0, 0)
}

MINIMUM_SIMILARITY = 0.5
dataset_embed = "dataset_embed/"

# Load database embedding
names = []
embeddings = []

for file in os.listdir(dataset_embed):
    if file.endswith(".npy"):
        person_name = os.path.splitext(file)[0]

        embedding = np.load(
            os.path.join(dataset_embed, file)
        )

        embedding = embedding / np.linalg.norm(embedding)

        names.append(person_name)
        embeddings.append(embedding)

if not embeddings:
    raise ValueError("Database embedding kosong")

embeddings = np.array(embeddings)

app = FaceAnalysis(
    name="buffalo_sc",
    providers=["CPUExecutionProvider"]
)

app.prepare(ctx_id=-1, det_size=(320, 320))

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    faces = app.get(frame)

    for face in faces:
        x1, y1, x2, y2 = face.bbox.astype(int)

        query_embedding = face.normed_embedding

        similarities = embeddings @ query_embedding

        best_idx = np.argmax(similarities)
        best_similarity = similarities[best_idx]

        if best_similarity >= MINIMUM_SIMILARITY:
            person_name = names[best_idx]
        else:
            person_name = "Unknown"

        color = dict_name.get(person_name, (255, 255, 255))

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        cv2.putText(
            frame,
            f"{person_name}: {best_similarity:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()