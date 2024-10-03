import cv2
from ultralytics import YOLO

# # load YOLOv8 pre-train model
# # we can change other model like yolov8n.pt
model = YOLO('yolov8x.pt')

# open video
video_path = 'runs/detect/predict3/try 1.avi'
cap = cv2.VideoCapture(video_path)

# get the video's weight height and fps
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# set the output video's path and encode type
output_path = 'video/output_with_car_count.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# car amount
car_count_total = 0
car_count_total2 = 0

# track cars
tracked_cars = {}

# Set the allowed bounding box movement distance, used to determine whether it is the same grid
# This threshold can be adjusted to control detection accuracy
iou_threshold = 0.9
confidence_threshold = 0.7

# red line y location
red_line_y = 250

def calculate_iou(box1, box2):
    # Calculate the intersection-over-union ratio (IoU) of two bounding boxes
    x1, y1, x2, y2 = box1
    x1_prime, y1_prime, x2_prime, y2_prime = box2

    # Calculate the cross area
    xi1 = max(x1, x1_prime)
    yi1 = max(y1, y1_prime)
    xi2 = min(x2, x2_prime)
    yi2 = min(y2, y2_prime)
    inter_area = max(0, xi2 - xi1 + 1) * max(0, yi2 - yi1 + 1)

    # Calculate union area
    box1_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    box2_area = (x2_prime - x1_prime + 1) * (y2_prime - y1_prime + 1)
    union_area = box1_area + box2_area - inter_area

    # Returns the intersection ratio
    return inter_area / union_area

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # use YOLOv8 model to predict
    results = model.predict(frame)
    new_tracked_cars = {}

    # draw the red line
    color_red = (0, 0, 255)
    thickness = 2
    cv2.line(frame, (0, red_line_y), (frame_width, red_line_y), color_red, thickness)

    # Traverse the detection results to determine whether it is a new car
    for result in results[0].boxes:
        class_id = int(result.cls)
        confidence = result.conf
        if model.names[class_id] in ['car'] and confidence > confidence_threshold:
            # Get the coordinates of the bounding box
            x1, y1, x2, y2 = result.xyxy[0].cpu().numpy().astype(int)

            # calculate IoU, and determine whether it is a new vehicle
            is_new_car = True
            for car_id, car_box in tracked_cars.items():
                iou = calculate_iou((x1, y1, x2, y2), car_box)
                if iou > iou_threshold:  # IoU 超过阈值，说明是同一辆车
                    new_tracked_cars[car_id] = (x1, y1, x2, y2)
                    is_new_car = False
                    break

            # if is new see is it pass the red line
            if is_new_car:
                if y2 > red_line_y:
                    car_count_total += 1
                    car_count_total2 += 1
                    new_tracked_cars[car_count_total] = (x1, y1, x2, y2)

    # update track car
    tracked_cars = new_tracked_cars

    # put "car count" text on the right top corner
    text = f'Total car count: {car_count_total}'
    text2 = f'Total car count crossed line: {car_count_total2}'
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color_text = (0, 255, 0)
    thickness = 2
    cv2.putText(frame, text, (frame_width - 500, 50), font, font_scale, color_text, thickness)
    cv2.putText(frame, text2, (frame_width - 700, 80), font, font_scale, color_text, thickness)

    # write into output video
    out.write(frame)

    # show the video (if need)
    cv2.imshow('YOLOv8 Detection with Car Count', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release source
cap.release()
out.release()
cv2.destroyAllWindows()
