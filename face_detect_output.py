import cv2
from mtcnn import MTCNN

detector = MTCNN()

input_path = '004_VID.mp4'
output_path = 'output_with_faces.mp4'

cap = cv2.VideoCapture(input_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    print(f"处理帧: {frame_idx}")
    frame_idx += 1

    results = detector.detect_faces(frame)

    for res in results:
        x, y, w, h = [int(v) for v in res['box']]
        x, y = max(0, x), max(0, y)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, '人脸', (x, max(y - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()
