import cv2
import mediapipe as mp
from enum import Enum
# from pydantic import BaseModel
#
# from fastapi import FastAPI
# class Item(BaseModel):
#     name: str
#     description: str | None = None
#     price: float
#     tax: float | None = None
# app = FastAPI()
# @app.post("/items/")
# async def create_item(item: Item):
#     return item
# Face Mesh
face_mesh = mp.solutions.face_mesh.FaceMesh()

# Image
image=cv2.imread("person.png")
rgb_img = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
height,width,_ = image.shape
print("Height, Width", height, width)
# Facial landmarks
result = face_mesh.process(rgb_img)
for facial_landmarks in result.multi_face_landmarks:
    for i in range(0,468):
        pt1 = facial_landmarks.landmark[i]
        x= int(pt1.x*width)
        y=int(pt1.y*height)
        print(pt1)
        cv2.circle(image, (x, y),1,(255, 133, 233),-1)

cv2.imshow("Image",image)

cv2.waitKey(0)
#