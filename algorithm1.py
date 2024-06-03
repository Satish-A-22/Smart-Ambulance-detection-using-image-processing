import cv2
from ultralytics import YOLO
import cvzone
import os
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import math
import time
import threading

def extract_features(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (64, 128))
    hog = cv2.HOGDescriptor()
    hog_features = hog.compute(img)
    return hog_features.flatten()

# Path to the folder containing images
folder_path = './train'

# List to store image paths and corresponding labels
image_paths = []
labels = []

# Iterate over images in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # Get image path
        image_path = os.path.join(folder_path, filename)
        # Extract features and append to the lists
        features = extract_features(image_path)
        image_paths.append(image_path)
        labels.append(filename.split('_')[0])  # Extract label from filename

# Convert lists to numpy arrays
labels = np.array(labels)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(image_paths, labels, test_size=0.2, random_state=42)

# Extract features for training set
X_train_features = np.array([extract_features(image) for image in X_train])

# Train a Support Vector Machine classifier
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train_features, y_train)

def match_image(captured_img_path):
    # Extract features from captured image
    captured_features = extract_features(captured_img_path)
    # Predict the class of the captured image
    predicted_label = svm_classifier.predict([captured_features])[0]
    # Check if there's a match in the training labels
    if predicted_label in y_train:
        matched_image_path = X_train[np.where(y_train == predicted_label)[0][0]]
        print("Matched image:", matched_image_path)
        # Prompt the user for input
        user_input = input("Object matched. Enter 'True' or 'False': ")
        if user_input.lower() == 'true':
            return True
        elif user_input.lower() == 'false':
            return False
        else:
            print("Invalid input. Defaulting to 'True' after 2 seconds.")
            # Default to True after 2 seconds
            timer = threading.Timer(2.0, lambda: True)
            timer.start()
            timer.join()
            return True
    else:
        # print("Object not detected.")
        return False

def camera():
    # For Webcam
    val=False
    # cap = cv2.VideoCapture('http://192.168.1.5:8080/video')
    # cap = cv2.VideoCapture('http://192.168.1.33:81/stream')
    cap = cv2.VideoCapture(1)
    cap.set(3, 720)
    cap.set(4, 720)
    model = YOLO("../Yolo-Weights/yolov8n.pt")

    classNames = ["person","ambulance", "bicycle", "motorbike", "aeroplane", "ambulance", "train", "truck", "boat",
                  "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                  "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                  "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                  "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                  "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                  "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                  "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                  "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                  "teddy bear", "hair drier", "toothbrush"
                  ]

    prev_frame_time = 0
    new_frame_time = 0

    # Create a folder to store captured images
    output_folder = "captured"
    os.makedirs(output_folder, exist_ok=True)
    i=1
    while True:
        new_frame_time = time.time()
        success, img = cap.read()
        results = model(img, stream=True)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Bounding Box
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                w, h = x2 - x1, y2 - y1
                cvzone.cornerRect(img, (x1, y1, w, h))
                # Confidence
                conf = math.ceil((box.conf[0] * 100)) / 100
                # Class Name
                cls = int(box.cls[0])
                cvzone.putTextRect(img, f'{classNames[cls]}   {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

                # Save captured image within the bounding box to the output folder
                cropped_img = img[y1:y2, x1:x2]
                cv2.imwrite(os.path.join(output_folder, f"{i}.jpg"), cropped_img)
                val = match_image(os.path.join(output_folder, f"{i}.jpg"))
                if val:
                    return True
                else:
                    return False
                i += 1

        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time

        cv2.imshow("Image", img)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

camera()
