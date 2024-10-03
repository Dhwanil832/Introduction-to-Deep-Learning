from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

video_path = "video/try 1.mkv"

model = YOLO("yolov8x.pt")
video_output = model.predict(source=video_path, conf=0.7,
                             save=True)