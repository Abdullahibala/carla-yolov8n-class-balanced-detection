from pathlib import Path
from collections import Counter
import cv2

# Dataset paths
BASE_DIR = Path(__file__).resolve().parent / "carla_dataset" / "labeled_dataset"
IMAGE_DIR = BASE_DIR / "images"
LABEL_DIR = BASE_DIR / "labels"

CLASS_NAMES = [
    "car",
    "pedestrian",
    "motorcycle",
    "bicycle",
    "traffic_light",
    "traffic_sign",
    "truck",
    "bus",
]

image_files = sorted(IMAGE_DIR.glob("*.png"))
label_files = sorted(LABEL_DIR.glob("*.txt"))

print("=" * 55)
print("YOLO DATASET VALIDATION")
print("=" * 55)

print(f"Images found: {len(image_files)}")
print(f"Label files found: {len(label_files)}")

image_stems = {p.stem for p in image_files}
label_stems = {p.stem for p in label_files}

print(f"Images without label files: {len(image_stems - label_stems)}")
print(f"Labels without images: {len(label_stems - image_stems)}")

class_counts = Counter()
empty_labels = []
invalid_lines = []
invalid_boxes = []
total_boxes = 0

for label_path in label_files:
    lines = [
        line.strip()
        for line in label_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    if not lines:
        empty_labels.append(label_path.name)
        continue

    for line_number, line in enumerate(lines, start=1):
        parts = line.split()

        if len(parts) != 5:
            invalid_lines.append(
                (label_path.name, line_number, line)
            )
            continue

        try:
            class_id = int(parts[0])
            x_center, y_center, width, height = map(
                float, parts[1:]
            )
        except ValueError:
            invalid_lines.append(
                (label_path.name, line_number, line)
            )
            continue

        if not 0 <= class_id < len(CLASS_NAMES):
            invalid_lines.append(
                (label_path.name, line_number, line)
            )
            continue

        values = [x_center, y_center, width, height]

        if not all(0.0 <= value <= 1.0 for value in values):
            invalid_boxes.append(
                (label_path.name, line_number, line)
            )
            continue

        if width <= 0 or height <= 0:
            invalid_boxes.append(
                (label_path.name, line_number, line)
            )
            continue

        class_counts[class_id] += 1
        total_boxes += 1

print(f"\nTotal bounding boxes: {total_boxes}")
print(f"Images with empty labels: {len(empty_labels)}")
print(f"Invalid label lines: {len(invalid_lines)}")
print(f"Invalid bounding boxes: {len(invalid_boxes)}")

print("\nOBJECT COUNTS BY CLASS")
for class_id, class_name in enumerate(CLASS_NAMES):
    print(f"{class_id} - {class_name}: {class_counts[class_id]}")

if total_boxes:
    print(f"\nAverage boxes per image: {total_boxes / max(len(image_files), 1):.2f}")

print("\nSAMPLE NON-EMPTY LABEL FILES")

shown = 0

for label_path in label_files:
    content = label_path.read_text(encoding="utf-8").strip()

    if content:
        print(f"\n{label_path.name}")
        print(content)

        image_path = IMAGE_DIR / f"{label_path.stem}.png"
        image = cv2.imread(str(image_path))

        if image is not None:
            print(f"Image dimensions: {image.shape[1]}x{image.shape[0]}")

        shown += 1

        if shown >= 3:
            break

print("\nValidation report complete.")