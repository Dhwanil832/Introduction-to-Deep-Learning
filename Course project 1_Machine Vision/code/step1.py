from ultralytics import YOLO
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# load pre-train model
# model = YOLO('yolov8n.pt')
model = YOLO('yolov8x.pt')