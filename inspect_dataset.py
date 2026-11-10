
import hashlib
from pathlib import Path

import cv2
import numpy as np

# Dataset directory
IMAGE_DIR = Path("carla_dataset/driving_images")

# Where to save inspection results
REPORT_DIR = Path("carla_dataset/inspection")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

CONTACT_SHEET_PATH = REPORT_DIR / "contact_sheet.jpg"

# Inspect up to 25 images
NUM_CONTACT_IMAGES = 25


def calculate_hash(image_path):
    """Calculate an exact file hash."""
    digest = hashlib.sha256()

    with open(image_path, "rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            digest.update(chunk)

    return digest.hexdigest()


def main():
    image_paths = sorted(IMAGE_DIR.glob("driving_*.png"))

    if not image_paths:
        print("No images found in:", IMAGE_DIR.resolve())
        return

    print("=" * 55)
    print("CARLA DATASET INSPECTION")
    print("=" * 55)

    print("Dataset directory:", IMAGE_DIR.resolve())
    print("Total images:", len(image_paths))

    hashes = {}
    dimensions = {}
    thumbnails = []

    previous_gray = None
    differences = []

    for index, image_path in enumerate(image_paths):
        file_hash = calculate_hash(image_path)

        hashes.setdefault(file_hash, []).append(image_path.name)

        image = cv2.imread(str(image_path))

        if image is None:
            print("WARNING: Cannot read:", image_path.name)
            continue

        height, width = image.shape[:2]
        dimensions[(width, height)] = (
            dimensions.get((width, height), 0) + 1
        )

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if previous_gray is not None:
            difference = np.mean(
                cv2.absdiff(previous_gray, gray)
            )

            differences.append(difference)

        previous_gray = gray

        # Select evenly spaced images for the contact sheet.
        if len(image_paths) <= NUM_CONTACT_IMAGES:
            select_image = True
        else:
            step = max(
                1,
                len(image_paths) // NUM_CONTACT_IMAGES
            )
            select_image = index % step == 0

        if select_image and len(thumbnails) < NUM_CONTACT_IMAGES:
            thumbnail = cv2.resize(
                image,
                (320, 180)
            )

            cv2.putText(
                thumbnail,
                image_path.name,
                (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

            thumbnails.append(thumbnail)

    duplicate_groups = [
        names for names in hashes.values()
        if len(names) > 1
    ]

    duplicate_image_count = sum(
        len(names) - 1 for names in duplicate_groups
    )

    print("\nIMAGE DIMENSIONS")
    for dimension, count in sorted(dimensions.items()):
        print(f"{dimension[0]}x{dimension[1]}: {count} images")

    print("\nDUPLICATE CHECK")
    print("Unique image contents:", len(hashes))
    print("Exact duplicate groups:", len(duplicate_groups))
    print("Extra duplicate files:", duplicate_image_count)

    if duplicate_groups:
        print("\nFirst duplicate groups:")
        for group in duplicate_groups[:10]:
            print(group)

    if differences:
        print("\nCONSECUTIVE IMAGE DIFFERENCES")
        print(f"Mean pixel difference: {np.mean(differences):.3f}")
        print(f"Minimum: {np.min(differences):.3f}")
        print(f"Maximum: {np.max(differences):.3f}")

    if thumbnails:
        columns = 5
        rows = (len(thumbnails) + columns - 1) // columns

        sheet = np.zeros(
            (rows * 180, columns * 320, 3),
            dtype=np.uint8
        )

        for i, thumbnail in enumerate(thumbnails):
            row = i // columns
            col = i % columns

            sheet[
                row * 180:(row + 1) * 180,
                col * 320:(col + 1) * 320
            ] = thumbnail

        cv2.imwrite(
            str(CONTACT_SHEET_PATH),
            sheet
        )

        print("\nContact sheet saved to:")
        print(CONTACT_SHEET_PATH.resolve())

    print("\nInspection complete.")


if __name__ == "__main__":
    main()
