import cv2

# 打开视频文件
video_path = 'runs/detect/predict3/try 1.avi'
cap = cv2.VideoCapture(video_path)

# 获取视频的宽度、高度和帧率
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# 设置输出视频的编码和输出路径
output_path = 'video/output_with_lines.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# 定义两条平行线的位置（在每一帧上绘制）
line1_start = (0, 450)  # 第一条线的起点
line1_end = (800, 600)    # 第一条线的终点
line2_start = (40, 350)  # 第二条线的起点
line2_end = (840, 500)    # 第二条线的终点

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 在每一帧上画两条平行线
    color1 = (255, 0, 0)
    color2 = (0, 0, 255)
    thickness = 2        # 线的厚度
    cv2.line(frame, line1_start, line1_end, color1, thickness)
    cv2.line(frame, line2_start, line2_end, color2, thickness)

    # 将处理后的帧写入输出视频
    out.write(frame)

    # cv2.imshow('Frame with Lines', frame)
    # if cv2.waitKey(100) & 0xFF == ord('q'):
    #     break

# release source
cap.release()
out.release()
cv2.destroyAllWindows()
