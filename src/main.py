import cv2
from tqdm import tqdm
from pathlib import Path
from detector import CrowdDetector


def process_video():
    # Детектор
    detector = CrowdDetector(model_path='models/yolov8x.pt', conf_thresh=0.4)

    input_path = Path('data/input/crowd.mp4')
    output_path = Path('data/output/processed_crowd.mp4')

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {input_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height))

    stats = []
    with tqdm(total=total_frames, desc="Processing video") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame, count = detector.detect(frame)
            writer.write(processed_frame)
            stats.append(count)
            pbar.update(1)

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    print(f"\nAverage people count: {sum(stats) / len(stats):.1f}")
    print(f"Max detected: {max(stats):.1f}")
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    process_video()