from ultralytics import YOLO
import matplotlib.pyplot as plt

if __name__ == "__main__":
    image_test_path = ("dataset/truck/truck_original_003.jpg_3aec88fc-d9f6-4705-b457-3df8a83be656.jpg")
    model = YOLO('yolov8s.pt')

    # set confidence like threshold
    results = model.predict(source=image_test_path, imgsz=640, conf=0.2)
    test_image = results[0].plot(line_width=2)
    plt.imshow(test_image)
    plt.show()