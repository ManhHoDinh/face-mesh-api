from typing import Annotated
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import cv2
import mediapipe as mp
import numpy as np  # Add this line to import NumPy
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
# Face Mesh
face_mesh = mp.solutions.face_mesh.FaceMesh()

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World 2"}

@app.post("/files/")
async def create_file(file: Annotated[bytes, File()]):
    return {"file_size": len(file)}

@app.post("/detectImage/")
async def create_upload_file(file: UploadFile):
    if not file.content_type.startswith('image'):
        return JSONResponse(content={"error": "Invalid file format"}, status_code=400)

    # Read the uploaded image file
    file_bytes = await file.read()
    image = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), -1)
    
    # Check if the image was read successfully
    if image is None:
        return JSONResponse(content={"error": "Unable to read image"}, status_code=400)

    # Convert the image to RGB
    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb_img = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    height,width,_ = image.shape
    print("Height, Width", height, width)
    # Facial landmarks
    result = face_mesh.process(rgb_img)
    results=[]
    if result.multi_face_landmarks:
        # Store the detected landmarks
        results = []
        for facial_landmarks in result.multi_face_landmarks:
            for i in range(0, 468):  # Assuming 468 landmarks
                pt1 = facial_landmarks.landmark[i]
                x = int(pt1.x * width)
                y = int(pt1.y * height)
                results.append({"x": x, "y": y})

        # Return the detected facial landmarks as JSON
        return JSONResponse(content=results)
    else:
        return JSONResponse(content={"error": "No face landmarks detected"}, status_code=400)
    #return Response(content=images, media_type="multipart/form-data")

def videoDetect(videoPath):
    cap = cv2.VideoCapture(videoPath)
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:
        while cap.isOpened():
            success, img = cap.read()
            image = cv2.resize(img, (1000, 1000))
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image)

            # Draw the face mesh annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_tesselation_style())
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_contours_style())
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_IRISES,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_iris_connections_style())
            # Flip the image horizontally for a selfie-view display.
            cv2.imshow('MediaPipe Face Mesh', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break
    cap.release()
    while (cap.isOpened()):

        # Capture frame-by-frame
        ret, frame = cap.read()
        image = cv2.resize(frame, (1000, 1000))
        if ret == True:
            # Display the resulting frame
            rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width, _ = image.shape
            # Facial landmarks
            result = face_mesh.process(rgb_img)
            if result.multi_face_landmarks:
                for face_landmarks in result.multi_face_landmarks:
                    for i in range(0, 468):
                        pt1 = face_landmarks.landmark[i]
                        x = int(pt1.x * width)
                        y = int(pt1.y * height)
                        print(pt1)
                        cv2.circle(image, (x, y), 1, (255, 133, 233), -1)

            cv2.imshow('MediaPipe Face Mesh', image)

            # Press Q on keyboard to exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break

    # When everything done, release
    # the video capture object
    cap.release()
    #     cv2.waitKey(0)
    cv2.destroyAllWindows()
