import cv2
from mtcnn import MTCNN

# 初始化人脸检测器（你也可以换成 cv2.CascadeClassifier）
detector = MTCNN()

# 读取视频
input_path = '004_VID.mp4'  # 输入视频路径
output_path = 'output_with_faces.mp4'  # 输出视频路径

cap = cv2.VideoCapture(input_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 编码格式
fps = cap.get(cv2.CAP_PROP_FPS)  # 视频帧率
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 初始化输出视频写入器
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 使用MTCNN检测人脸
    results = detector.detect_faces(frame)

    for res in results:
        x, y, w, h = res['box']
        # 画人脸框
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # 在人脸框上方写“人脸”两个字
        cv2.putText(frame, '人脸', (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # 写入处理后的帧
    out.write(frame)

# 释放资源
cap.release()
out.release()
cv2.destroyAllWindows()
