# from ultralytics import YOLO
# import cv2
#
# model = YOLO('../Yolo-Weights/yolov8l.pt')
# results = model("./img/ambulance.jpg", show=True)
# cv2.waitKey

import cv2
import os
from ultralytics import YOLO
import cvzone
import math
import time
# import algorithm as pg3
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf

TF_ENABLE_ONEDNN_OPTS = 0

def camera():
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

    while True:
        new_frame_time = time.time()
        success, img = cap.read()
        results = model(img, stream=True)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Bounding Box
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1

                # Confidence
                conf = math.ceil((box.conf[0] * 100)) / 100
                # Class Name
                cls = int(box.cls[0])
                class_name = classNames[cls]

                # Draw bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                cvzone.cornerRect(img, (x1, y1, w, h))
                cvzone.putTextRect(img, f'{class_name}   {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)
                print(class_name)
                # Save captured image if it's an ambulance
                if class_name == "ambulance" or class_name=='remote' or classNames=='truck' or class_name=='cell phone':
                    cropped_img = img[y1:y2, x1:x2]
                    cv2.imwrite("captured/ambulance.jpg", cropped_img)
                    # print("True")
                    return True
                else:
                    # print("False")
                    return False

        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time

        cv2.imshow("Image", img)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

camera()

