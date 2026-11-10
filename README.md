# CARLA Object Detection Dataset Tools

Code and collection notes for a master's thesis project on efficient traffic object detection with YOLOv8n. The project uses CARLA 0.9.16 to generate YOLO-format annotations and study class coverage for cars, pedestrians, motorcycles, bicycles, traffic lights, traffic signs, trucks, and buses.

## Current progress

- CARLA client and simulator setup confirmed on Windows.
- Data collection, actor inspection, annotation validation, and label visualization scripts are available here.
- The latest balanced collection run reported 90 images and 1,056 boxes, including all eight target classes. These figures are from the collection report and still require manual annotation quality review.
- Dataset files and rendered visualizations are intentionally excluded from this repository. Keep the dataset separately and do not commit generated images or labels.
- Model training, controlled baseline/balanced comparison, and statistical evaluation have not yet been completed.

## Requirements

- Windows and Python 3.12 (the current development environment)
- CARLA 0.9.16 server and its matching Python API available to the active Python interpreter
- Python packages listed in [`requirements.txt`](requirements.txt)

Install the Python packages:

```powershell
python -m pip install -r requirements.txt
```

Start CARLA 0.9.16 with Town10HD_Opt available, then from this directory run:

```powershell
python test_carla.py
python collect_balanced_data.py
```

The balanced collector currently saves beneath `carla_dataset/balanced_dataset/`. Read its configuration at the top of the script before changing capture settings. Other scripts are earlier collection and inspection utilities; review their path/configuration constants before running them.

## Dataset utilities

```powershell
python validate_labels.py
python visualize_labels.py
python visualize_labels.py --all
```

The visualizer writes rendered images under `carla_dataset/balanced_dataset/visualized/`. The validator currently targets the original `labeled_dataset` directory; check its constants if you want to validate a different dataset folder.

## Dataset format

Images and labels are paired by filename. Each YOLO label row is:

```text
class_id x_center y_center width height
```

Coordinates are normalized to image width and height. Class order is defined in the collector and recorded in `classes.txt` alongside locally collected data. Preserve the same class order across collection and training.

## Research caveats

The CARLA labels are generated automatically. A valid YOLO row does not guarantee that the box is visible, correctly classified, or semantically appropriate. In particular, static map bounding boxes can include sign-like advertising surfaces, and projected actor boxes may include occluded objects. Inspect overlays and correct/ filter labels before using them for experiments. Split data by collection run or scene where practical to reduce near-duplicate train/test leakage. Simulated results should be reported as CARLA results and not assumed to establish real-road performance.

## Repository contents

- `collect_balanced_data.py` — targeted balanced collection, including rare vehicle views and map sign annotations.
- `collect_labeled_data.py`, `collect_driving_data.py`, `capture_images.py` — earlier collection experiments.
- `inspect_carla_actors.py`, `inspect_dataset.py` — actor and dataset inspection.
- `validate_labels.py`, `visualize_labels.py` — annotation checks and overlay rendering.
- `spawn_vehicle.py`, `test_carla.py` — basic CARLA connectivity and actor smoke checks.

## License

No license has been selected yet. Until one is added, reuse and redistribution are not granted by this repository.
