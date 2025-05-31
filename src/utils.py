import cv2


def draw_detections(frame, results):
    """Отрисовка bounding boxes на кадре"""
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf.item()

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Подпись
            label = f"person: {conf:.2f}"
            cv2.putText(
                frame, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
            )
    return frame