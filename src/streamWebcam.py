import cv2
import numpy as np
import requests
import time

# Define the API endpoint
api_url = "http://127.0.0.1:5000/predict"

# Load face detector
face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

# Start video capture
cap = cv2.VideoCapture(0)

# Initialize FPS variables
fps = 0
prev_time = time.time()

while True:
    # Capture frame from the webcam
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1280, 720))
    if not ret:
        break

    # Start time for FPS calculation
    start_time = time.time()

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in num_faces:
        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]

        # Preprocess the face image
        cropped_img = cv2.resize(roi_gray_frame, (48, 48))
        is_success, im_buf_arr = cv2.imencode(".jpg", cropped_img)
        byte_im = im_buf_arr.tobytes()

        # Send the image to the API
        response = requests.post(api_url, files={"file": byte_im})

        # Check if the API returned a valid response
        if response.status_code == 200:
            emotion = response.json().get("emotion")
            # Display the emotion label on the frame
            cv2.putText(frame, emotion, (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        else:
            print("Error:", response.status_code, response.text)

    # Calculate and display FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Display the resulting frame
    cv2.imshow('Emotion Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
