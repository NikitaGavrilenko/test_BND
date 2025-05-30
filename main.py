import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
from deep_sort_realtime.deepsort_tracker import DeepSort
import argparse


def detect_people(frame, model, tracker, conf_threshold=0.4):
    """
    Детекция людей на кадре с использованием YOLO и DeepSORT

    Args:
        frame: Входной кадр видео
        model: Загруженная модель YOLO
        tracker: Трекер DeepSORT
        conf_threshold: Порог уверенности для детекции

    Returns:
        Обработанный кадр с bounding boxes
        Список детекций для анализа
    """
    # Детекция с использованием YOLO
    results = model(frame, classes=[0], conf=conf_threshold, verbose=False, agnostic_nms=True)

    # Преобразование результатов для DeepSORT
    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf.item()
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, 'person'))

    # Обновление трекера с обработкой ошибок
    try:
        tracks = tracker.update_tracks(detections, frame=frame)
    except Exception as e:
        print(f"Ошибка трекинга: {str(e)}")
        tracks = []

    # Визуализация результатов
    analyzed_detections = []
    for track in tracks:
        if not track.is_confirmed():
            continue

        try:
            track_id = str(track.track_id) if hasattr(track, 'track_id') else "0"
            ltrb = track.to_ltrb() if hasattr(track, 'to_ltrb') else [0, 0, 10, 10]
            x1, y1, x2, y2 = map(int, ltrb)

            # Безопасное получение confidence
            confidence = 0.0
            if hasattr(track, 'get_det_conf'):
                try:
                    confidence = float(track.get_det_conf())
                except:
                    confidence = 0.0

            # Генерация цвета на основе ID трека
            try:
                color_hash = hash(track_id) if track_id else 0
                color = (
                    int(color_hash % 255),
                    int((color_hash * 2) % 255),
                    int((color_hash * 3) % 255)
                )
            except:
                color = (0, 255, 0)  # Зеленый по умолчанию

            # Анализ размера бокса
            box_area = (x2 - x1) * (y2 - y1)
            analyzed_detections.append({
                'track_id': track_id,
                'bbox': (x1, y1, x2, y2),
                'confidence': confidence,
                'box_area': box_area
            })

            # Отрисовка с цветом по ID трека
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Безопасное создание подписи
            label = f"ID:{track_id}"
            if confidence > 0:
                label += f" conf:{confidence:.2f}"

            cv2.putText(frame, label, (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        except Exception as e:
            print(f"Ошибка обработки трека (кадр {len(analyzed_detections)}): {str(e)}")
            continue

    return frame, analyzed_detections
def enhance_image(frame):
    """
    Улучшение качества изображения для лучшей детекции

    Args:
        frame: Входной кадр

    Returns:
        Улучшенный кадр
    """
    # Контрастная адаптивная гистограммная эквализация
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced_lab = cv2.merge((cl, a, b))
    enhanced_frame = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    # Легкое повышение резкости
    kernel = np.array([[-1, -1, -1],
                       [-1, 9, -1],
                       [-1, -1, -1]])
    return cv2.filter2D(enhanced_frame, -1, kernel)


def main():
    parser = argparse.ArgumentParser(description='Crowd Detection System')
    parser.add_argument('--input', default='crowd.mp4', help='Input video file')
    parser.add_argument('--output', default='crowd_output.mp4', help='Output video file')
    parser.add_argument('--model', default='yolov8x.pt', help='YOLO model file')
    parser.add_argument('--conf', type=float, default=0.4, help='Confidence threshold')
    parser.add_argument('--enhance', action='store_true', help='Enable image enhancement')
    args = parser.parse_args()

    # Загрузка более мощной модели
    print(f"Загрузка модели {args.model}...")
    model = YOLO(args.model)

    # Инициализация трекера DeepSORT
    tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0)

    # Открытие видеофайла
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print("Ошибка открытия видеофайла")
        return

    # Получение параметров видео
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Создание VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    # Статистика для анализа
    detection_stats = []
    frame_counter = 0

    # Обработка видео с прогресс-баром
    for _ in tqdm(range(total_frames), desc="Обработка видео"):
        ret, frame = cap.read()
        if not ret:
            break

        frame_counter += 1

        # Улучшение изображения при необходимости
        if args.enhance:
            frame = enhance_image(frame)

        # Детекция и трекинг людей
        processed_frame, detections = detect_people(frame, model, tracker, args.conf)

        # Сохранение статистики
        detection_stats.append({
            'frame': frame_counter,
            'detections': detections,
            'count': len(detections)
        })

        # Отображение счетчика объектов
        count_text = f"People: {len(detections)}"
        cv2.putText(processed_frame, count_text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Запись кадра
        out.write(processed_frame)

    # Освобождение ресурсов
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Анализ результатов
    analyze_results(detection_stats)
    print(f"Обработка завершена. Результат сохранен в {args.output}")


def analyze_results(stats):
    """
    Анализ результатов детекции

    Args:
        stats: Собранная статистика по детекции
    """
    total_detections = sum(frame['count'] for frame in stats)
    avg_detections = total_detections / len(stats)

    print("\n" + "=" * 50)
    print("АНАЛИЗ РЕЗУЛЬТАТОВ ДЕТЕКЦИИ")
    print("=" * 50)
    print(f"Всего кадров: {len(stats)}")
    print(f"Общее количество детекций: {total_detections}")
    print(f"Среднее количество людей на кадр: {avg_detections:.2f}")

    # Анализ стабильности трекинга
    track_ids = set()
    for frame in stats:
        for det in frame['detections']:
            track_ids.add(det['track_id'])

    print(f"Уникальных треков: {len(track_ids)}")

    # Рекомендации по улучшению
    print("\nРЕКОМЕНДАЦИИ ПО УЛУЧШЕНИЮ:")
    if avg_detections < 10:
        print("- Используйте более мощную модель (yolov8x или yolov9)")
        print("- Уменьшите порог уверенности (--conf 0.3)")
    else:
        print("- Для обработки толпы попробуйте модель с лучшей производительностью на плотных сценах")
        print("- Добавьте кластеризацию мелких объектов")

    print("- Для сложных условий освещения включите улучшение изображения (--enhance)")
    print("- Рассмотрите возможность использования ансамбля моделей для повышения точности")


if __name__ == "__main__":
    main()