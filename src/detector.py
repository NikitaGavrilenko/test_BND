import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from pathlib import Path


class CrowdDetector:
    def __init__(self, model_path='models/yolov8x.pt', conf_thresh=0.4):
        self.model = YOLO(model_path)
        self.tracker = DeepSort(max_age=30, n_init=3)
        self.conf_thresh = conf_thresh
        self.track_colors = {}

    def enhance_image(self, frame):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        enhanced = cv2.merge((cl, a, b))
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    def detect(self, frame):
        # Детекция объектов
        results = self.model(
            frame,
            classes=[0],
            conf=self.conf_thresh,
            imgsz=640,
            verbose=False
        )

        # Подготовка данных для трекера
        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf.item()
                detections.append(([x1, y1, x2 - x1, y2 - y1], conf, 'person'))

        # Трекинг объектов
        tracks = self.tracker.update_tracks(detections, frame=frame)

        # Визуализация результатов
        people_count = 0
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = str(track.track_id)
            x1, y1, x2, y2 = map(int, track.to_ltrb())

            # Безопасное получение confidence
            conf = getattr(track, 'get_det_conf', lambda: self.conf_thresh)()
            if conf is None:
                conf = self.conf_thresh

            # Генерация цвета
            color = self._get_color(track_id)

            # Отрисовка
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"ID:{track_id}"
            if conf is not None:
                label += f" {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            people_count += 1

        return frame, people_count

    def _get_color(self, track_id):
        if track_id not in self.track_colors:
            self.track_colors[track_id] = (
                int(hash(track_id) % 256),
                    int(hash(track_id + "a") % 256),
                        int(hash(track_id + "b") % 256))
        return self.track_colors[track_id]