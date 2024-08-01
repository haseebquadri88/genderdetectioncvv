from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import cvlib as cv

# Load the pre-trained gender detection model
model = load_model('gender_detection.model.keras')

# Open the webcam
webcam = cv2.VideoCapture(0)
if not webcam.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Define the class labels
classes = ['male', 'female']

# Start capturing video frames
while webcam.isOpened():
    # Capture frame-by-frame
    ret, frame = webcam.read()

    if not ret:
        break

    # Detect faces in the frame
    faces, confidences = cv.detect_face(frame)

    # Process each detected face
    for face in faces:
        (startX, startY, endX, endY) = face

        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

        # Crop the detected face region
        face_crop = frame[startY:endY, startX:endX]

        # Skip faces that are too small
        if face_crop.shape[0] < 10 or face_crop.shape[1] < 10:
            continue

        # Preprocess the face for the model
        face_crop = cv2.resize(face_crop, (96, 96))
        face_crop = face_crop.astype("float32") / 255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)

        # Predict the gender
        predictions = model.predict(face_crop)[0]
        label_index = np.argmax(predictions)
        label = classes[label_index]

        # Format the label with confidence
        confidence = predictions[label_index] * 100
        label_text = f"{label}: {confidence:.2f}%"

        # Position the label above the rectangle
        label_y = max(startY - 10, 10)
        cv2.putText(frame, label_text, (startX, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow("Gender Detection", frame)

    # Exit loop on 'Q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources and close windows
webcam.release()
cv2.destroyAllWindows()
