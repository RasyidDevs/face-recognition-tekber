import cv2
import os
import time
import numpy as np
from insightface.app import FaceAnalysis
app = FaceAnalysis(name="buffalo_sc", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
dataset_path = "dataset/"
dataset_embed = "dataset_embed/"
if not os.path.exists(dataset_embed):
    os.makedirs(dataset_embed)
app.prepare(ctx_id=-1, det_size=(320, 320))
for person_name in os.listdir(dataset_path):
    if person_name in [os.path.splitext(f)[0] for f in os.listdir(dataset_embed) if f.endswith(".npy")]:
        print(f"Embedding for {person_name} already exists. Skipping...")
        continue
    start_time = time.time()
    embeddings = []
    for image in os.listdir(dataset_path + person_name):
        img = cv2.imread(dataset_path + person_name + "/" + image)
        faces = app.get(img)
        for face in faces:
            embedding = face.normed_embedding
            embeddings.append(embedding)
    pooled = np.mean(embeddings, axis=0)
    pooled = pooled / np.linalg.norm(pooled)
    np.save(dataset_embed + person_name + ".npy", pooled)
    end_time = time.time()
    print(f"Processed {person_name} in {end_time - start_time:.2f} seconds. Embedding shape: {pooled.shape}")