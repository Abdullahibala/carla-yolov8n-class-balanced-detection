"""Render YOLO labels from the latest balanced-dataset collection run."""

import argparse
import math
import re
from collections import Counter
from pathlib import Path

import cv2


BASE_DIR = (
    Path(__file__).resolve().parent
    / "carla_dataset"
    / "balanced_dataset"
)
IMAGE_DIR = BASE_DIR / "images"
LABEL_DIR = BASE_DIR / "labels"
OUTPUT_DIR = BASE_DIR / "visualized"

FALLBACK_CLASS_NAMES = [
    "car",
    "pedestrian",
    "motorcycle",
    "bicycle",
    "traffic_light",
    "traffic_sign",
    "truck",
    "bus",
]

# Distinct BGR colors make adjacent class boxes easier to tell apart.
CLASS_COLORS = [
    (60, 220, 60),
    (0, 200, 255),
    (255, 120, 0),
    (255, 0, 220),
    (0, 230, 230),
    (255, 180, 0),
    (0, 120, 255),
    (220, 60, 220),
]


def load_class_names():
    classes_path = BASE_DIR / "classes.txt"
    if classes_path.exists():
        names = [line.strip() for line in classes_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if names:
            return names
    return FALLBACK_CLASS_NAMES


def latest_run_images(images):
    """Use Saved images from the report to identify the newest appended run."""
    report_path = BASE_DIR / "dataset_report.txt"
    if not report_path.exists():
        return images

    report = report_path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"^Saved images:\s*(\d+)\s*$", report, re.MULTILINE)
    if not match:
        return images

    run_size = int(match.group(1))
    if run_size <= 0 or run_size > len(images):
        return images
    return images[-run_size:]


def parse_yolo_label(label_path, class_count):
    """Return valid (class_id, x, y, width, height) rows; report bad rows."""
    if not label_path.exists():
        return None

    parsed = []
    for line_number, line in enumerate(
        label_path.read_text(encoding="utf-8", errors="replace").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue

        parts = line.split()
        try:
            if len(parts) != 5:
                raise ValueError("expected five columns")
            class_id = int(parts[0])
            values = tuple(float(value) for value in parts[1:])
            if not 0 <= class_id < class_count:
                raise ValueError(f"class id {class_id} is outside the class list")
            if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in values):
                raise ValueError("coordinates must be finite and normalized to [0, 1]")
            _, _, box_width, box_height = values
            if box_width <= 0.0 or box_height <= 0.0:
                raise ValueError("box width and height must be positive")
            parsed.append((class_id, *values))
        except ValueError as exc:
            print(f"Invalid label {label_path.name}:{line_number}: {exc}")

    return parsed


def choose_samples(images, labels_by_stem, sample_count, class_count):
    """Pick frames that cover rare classes, then spread remaining samples out."""
    if sample_count >= len(images):
        return images

    rows_by_stem = {
        stem: rows
        for stem, rows in labels_by_stem.items()
        if rows is not None
    }
    class_image_counts = Counter(
        class_id
        for rows in rows_by_stem.values()
        for class_id in {row[0] for row in rows}
    )
    images_by_class = {
        class_id: [
            image
            for image in images
            if class_id in {row[0] for row in rows_by_stem.get(image.stem, [])}
        ]
        for class_id in range(class_count)
    }

    selected = []
    # Prefer one clear example from each class, starting with the rarest.
    for class_id, _ in sorted(class_image_counts.items(), key=lambda item: item[1]):
        candidates = images_by_class[class_id]
        if not candidates or len(selected) >= sample_count:
            continue
        best = max(
            candidates,
            key=lambda image: sum(
                1 for row in rows_by_stem.get(image.stem, []) if row[0] == class_id
            ),
        )
        if best not in selected:
            selected.append(best)

    # Fill the rest with evenly spaced frames from the current run.
    if len(selected) < sample_count:
        step = max(1, len(images) // sample_count)
        for image in images[::step]:
            if image not in selected:
                selected.append(image)
            if len(selected) >= sample_count:
                break

    if len(selected) < sample_count:
        for image in images:
            if image not in selected:
                selected.append(image)
            if len(selected) >= sample_count:
                break

    return sorted(selected[:sample_count], key=lambda path: path.name)


def draw_labels(image_path, label_path, output_path, class_names, rows):
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Could not read image: {image_path.name}")
        return False

    height, width = image.shape[:2]
    for class_id, x_center, y_center, box_width, box_height in rows:
        x1 = round((x_center - box_width / 2.0) * width)
        y1 = round((y_center - box_height / 2.0) * height)
        x2 = round((x_center + box_width / 2.0) * width)
        y2 = round((y_center + box_height / 2.0) * height)

        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(0, min(width - 1, x2))
        y2 = max(0, min(height - 1, y2))
        if x2 <= x1 or y2 <= y1:
            print(f"Skipping zero-area box in {label_path.name}")
            continue

        color = CLASS_COLORS[class_id % len(CLASS_COLORS)]
        name = class_names[class_id]
        text = f"{name} ({class_id})"
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        (text_width, text_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        text_top = max(0, y1 - text_height - baseline - 4)
        cv2.rectangle(
            image,
            (x1, text_top),
            (min(width - 1, x1 + text_width + 4), y1),
            color,
            thickness=-1,
        )
        cv2.putText(
            image,
            text,
            (x1 + 2, max(text_height, y1 - baseline - 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    try:
        if cv2.imwrite(str(output_path), image):
            return True
    except cv2.error as exc:
        print(f"OpenCV could not write {output_path.name}: {exc}")

    # OpenCV's file writer can fail transiently on Windows even when the
    # image encoder is available. Encode in memory and let Python write bytes.
    try:
        encoded, png_bytes = cv2.imencode(".png", image)
        if not encoded:
            print(f"Could not encode visualization: {output_path}")
            return False
        output_path.write_bytes(png_bytes.tobytes())
    except (cv2.error, OSError) as exc:
        print(f"Could not write visualization {output_path}: {exc}")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Visualize YOLO labels in the latest balanced CARLA dataset run."
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Visualize every image in the balanced_dataset folder, including older runs.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=12,
        help="Number of class-covering sample images to render (default: 12).",
    )
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be at least 1")

    if not IMAGE_DIR.exists() or not LABEL_DIR.exists():
        raise SystemExit(f"Balanced dataset images/labels folders not found under {BASE_DIR}")

    class_names = load_class_names()
    images = sorted(IMAGE_DIR.glob("pilot_*.png"))
    if not images:
        raise SystemExit(f"No pilot_*.png images found in {IMAGE_DIR}")

    latest_images = images if args.all else latest_run_images(images)
    labels_by_stem = {
        image.stem: parse_yolo_label(
            LABEL_DIR / f"{image.stem}.txt", len(class_names)
        )
        for image in latest_images
    }
    missing_labels = [stem for stem, rows in labels_by_stem.items() if rows is None]
    if missing_labels:
        print(f"Warning: {len(missing_labels)} images have no matching label file.")

    selected = (
        latest_images
        if args.all
        else choose_samples(latest_images, labels_by_stem, args.count, len(class_names))
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rendered = 0
    for image_path in selected:
        label_path = LABEL_DIR / f"{image_path.stem}.txt"
        rows = labels_by_stem.get(image_path.stem)
        if rows is None:
            print(f"Missing label: {label_path.name}")
            continue

        output_path = OUTPUT_DIR / f"{image_path.stem}.png"
        if draw_labels(image_path, label_path, output_path, class_names, rows):
            rendered += 1

    scope = "all dataset" if args.all else "latest-run"
    print(
        f"Rendered {rendered}/{len(selected)} selected images from "
        f"{len(latest_images)} {scope} images."
    )
    print(f"Output folder: {OUTPUT_DIR}")
    if args.all:
        print("Note: --all included earlier runs that may use different annotation logic.")


if __name__ == "__main__":
    main()
