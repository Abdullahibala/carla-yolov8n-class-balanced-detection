import carla
import cv2
import math
import numpy as np
import queue
import random
import re
import time
from collections import Counter
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

HOST = "127.0.0.1"
PORT = 2000

# Start with a small pilot.
NUM_IMAGES = 50

IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720
CAMERA_FOV = 90.0

FPS = 20
FIXED_DELTA_SECONDS = 1.0 / FPS

NUM_TRAFFIC_VEHICLES = 30
NUM_WALKERS = 10

# Spread pilot images across a longer drive instead of sampling 50
# consecutive frames from the same blocked viewpoint.
CAPTURE_INTERVAL_FRAMES = 10
TARGETED_VIEWS_PER_TARGET_ACTOR = 5
MIN_VISIBLE_SIGN_PIXELS = 4

MAX_OBJECT_DISTANCE = 60.0
MIN_BOX_AREA = 25.0
MIN_SIGN_BOX_AREA = 25.0

RANDOM_SEED = 42

# New output directory. Existing data is not overwritten.
OUTPUT_DIR = (
    Path(__file__).resolve().parent
    / "carla_dataset"
    / "balanced_dataset"
)

IMAGE_DIR = OUTPUT_DIR / "images"
LABEL_DIR = OUTPUT_DIR / "labels"

CLASS_NAMES = [
    "car",           # 0
    "pedestrian",    # 1
    "motorcycle",    # 2
    "bicycle",       # 3
    "traffic_light", # 4
    "traffic_sign",  # 5
    "truck",         # 6
    "bus",           # 7
]

# Explicit vehicle classification.
MOTORCYCLES = {
    "vehicle.harley-davidson.low_rider",
    "vehicle.kawasaki.ninja",
    "vehicle.vespa.zx125",
    "vehicle.yamaha.yzf",
}

BICYCLES = {
    "vehicle.bh.crossbike",
    "vehicle.diamondback.century",
    "vehicle.gazelle.omafiets",
}

TRUCKS = {
    "vehicle.carlamotors.carlacola",
    "vehicle.carlamotors.european_hgv",
    "vehicle.carlamotors.firetruck",
    "vehicle.tesla.cybertruck",
}

BUSES = {
    "vehicle.mitsubishi.fusorosa",
}


# ============================================================
# INITIALIZATION
# ============================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

IMAGE_DIR.mkdir(parents=True, exist_ok=True)
LABEL_DIR.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_DIR / "classes.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(CLASS_NAMES))


# ============================================================
# CLASS MAPPING
# ============================================================

def get_class_id(actor):
    type_id = actor.type_id.lower()

    if type_id.startswith("walker.pedestrian"):
        return 1

    if type_id == "traffic.traffic_light":
        return 4

    if type_id in {"traffic.stop", "traffic.yield"}:
        return 5

    if type_id.startswith("vehicle."):
        if type_id in MOTORCYCLES:
            return 2

        if type_id in BICYCLES:
            return 3

        if type_id in TRUCKS:
            return 6

        if type_id in BUSES:
            return 7

        # Remaining vehicles are treated as cars for this pilot.
        return 0

    return None


# ============================================================
# CAMERA PROJECTION
# ============================================================

def build_camera_matrix(width, height, fov):
    focal = width / (2.0 * np.tan(np.radians(fov) / 2.0))

    return np.array([
        [focal, 0.0, width / 2.0],
        [0.0, focal, height / 2.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def world_to_camera(point, inverse_matrix):
    point_h = np.array(
        [point.x, point.y, point.z, 1.0],
        dtype=np.float64,
    )

    local = inverse_matrix @ point_h

    # CARLA camera axes: forward, right, up.
    # Pinhole camera axes: right, down, forward.
    return np.array(
        [local[1], -local[2], local[0]],
        dtype=np.float64,
    )


def project_bounding_box(
    bounding_box,
    object_transform,
    camera_transform,
    camera_matrix,
    min_box_area=MIN_BOX_AREA,
):
    """
    Project a CARLA bounding box into image coordinates.

    For map-level boxes, pass carla.Transform() because their
    locations and rotations are already in world coordinates.
    """

    try:
        vertices = bounding_box.get_world_vertices(object_transform)
    except (RuntimeError, AttributeError):
        return None

    inverse_matrix = np.asarray(
        camera_transform.get_inverse_matrix(),
        dtype=np.float64,
    )

    camera_points = []

    for vertex in vertices:
        point = world_to_camera(vertex, inverse_matrix)
        camera_points.append(point)

    camera_points = np.asarray(camera_points)

    # Keep points in front of the camera.
    front_points = camera_points[camera_points[:, 2] > 0.1]

    if len(front_points) < 3:
        return None

    projected = []

    for point in front_points:
        pixel = camera_matrix @ point
        u = pixel[0] / pixel[2]
        v = pixel[1] / pixel[2]
        projected.append([u, v])

    projected = np.asarray(projected)

    x_min = projected[:, 0].min()
    y_min = projected[:, 1].min()
    x_max = projected[:, 0].max()
    y_max = projected[:, 1].max()

    if (
        x_max <= 0
        or y_max <= 0
        or x_min >= IMAGE_WIDTH
        or y_min >= IMAGE_HEIGHT
    ):
        return None

    x_min = np.clip(x_min, 0, IMAGE_WIDTH - 1)
    y_min = np.clip(y_min, 0, IMAGE_HEIGHT - 1)
    x_max = np.clip(x_max, 0, IMAGE_WIDTH - 1)
    y_max = np.clip(y_max, 0, IMAGE_HEIGHT - 1)

    width = x_max - x_min
    height = y_max - y_min

    if width <= 0 or height <= 0:
        return None

    if width * height < min_box_area:
        return None

    return x_min, y_min, x_max, y_max


def project_actor_bbox(actor, actor_transform, camera_transform, camera_matrix):
    return project_bounding_box(
        actor.bounding_box,
        actor_transform,
        camera_transform,
        camera_matrix,
    )


def get_traffic_sign_bounding_boxes(world):
    """Return static sign boxes supplied by the loaded CARLA map."""
    try:
        return list(world.get_level_bbs(carla.CityObjectLabel.TrafficSigns))
    except (AttributeError, RuntimeError) as exc:
        print(f"Could not load map traffic-sign boxes: {exc}")
        return []


def get_semantic_tag_mask(image, tag_id):
    """Decode CARLA's raw semantic tag from the BGRA red channel."""
    pixels = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(
        IMAGE_HEIGHT, IMAGE_WIDTH, 4
    )
    return pixels[:, :, 2] == tag_id


def wait_for_sensor_frame(sensor_queue, frame_id, description):
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        try:
            image = sensor_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        if image.frame == frame_id:
            return image
        if image.frame > frame_id:
            break
    print(f"Skipped frame {frame_id}: {description} frame missing.")
    return None


def build_frame_labels(
    world,
    snapshot,
    camera,
    camera_matrix,
    ego_vehicle,
    traffic_sign_boxes,
    sign_mask,
):
    camera_pose = camera.get_transform()
    camera_location = camera_pose.location
    labels = []
    class_counts = Counter()

    for actor in world.get_actors():
        if not actor.is_alive or actor.id == ego_vehicle.id:
            continue

        class_id = get_class_id(actor)
        if class_id is None:
            continue

        actor_state = snapshot.find(actor.id)
        if actor_state is None:
            continue

        actor_transform = actor_state.get_transform()
        if actor_transform.location.distance(camera_location) > MAX_OBJECT_DISTANCE:
            continue

        box = project_actor_bbox(
            actor, actor_transform, camera_pose, camera_matrix
        )
        if box is None:
            continue

        labels.append(to_yolo_format(class_id, box))
        class_counts[class_id] += 1

    # Map-level boxes include signs behind vehicles and buildings. Require
    # visible traffic-sign pixels inside each projected box before labeling.
    for sign_box in traffic_sign_boxes:
        if sign_box.location.distance(camera_location) > MAX_OBJECT_DISTANCE:
            continue

        box = project_bounding_box(
            sign_box,
            carla.Transform(),
            camera_pose,
            camera_matrix,
            min_box_area=MIN_SIGN_BOX_AREA,
        )
        if box is None:
            continue

        x0, y0, x1, y1 = box
        left = max(0, int(math.floor(x0)))
        top = max(0, int(math.floor(y0)))
        right = min(IMAGE_WIDTH, int(math.ceil(x1)))
        bottom = min(IMAGE_HEIGHT, int(math.ceil(y1)))
        if right <= left or bottom <= top:
            continue

        if np.count_nonzero(sign_mask[top:bottom, left:right]) < MIN_VISIBLE_SIGN_PIXELS:
            continue

        labels.append(to_yolo_format(5, box))
        class_counts[5] += 1

    return labels, class_counts


def save_frame(image, labels, filename, image_counts, class_counts):
    image_array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(
        IMAGE_HEIGHT, IMAGE_WIDTH, 4
    )
    image_bgr = image_array[:, :, :3].copy()
    image_path = IMAGE_DIR / f"{filename}.png"
    label_path = LABEL_DIR / f"{filename}.txt"

    if not cv2.imwrite(str(image_path), image_bgr):
        raise RuntimeError(f"Failed to save {image_path}")

    with open(label_path, "w", encoding="utf-8") as label_file:
        label_file.write("\n".join(labels))

    class_counts.update(image_counts)
    return len(labels) == 0


def to_yolo_format(class_id, box):
    x_min, y_min, x_max, y_max = box

    x_center = ((x_min + x_max) / 2.0) / IMAGE_WIDTH
    y_center = ((y_min + y_max) / 2.0) / IMAGE_HEIGHT

    width = (x_max - x_min) / IMAGE_WIDTH
    height = (y_max - y_min) / IMAGE_HEIGHT

    return (
        f"{class_id} "
        f"{x_center:.6f} "
        f"{y_center:.6f} "
        f"{width:.6f} "
        f"{height:.6f}"
    )


# ============================================================
# VEHICLE SPAWNING
# ============================================================

def spawn_traffic(world, traffic_manager, ego_vehicle):
    blueprint_library = world.get_blueprint_library()
    ego_location = ego_vehicle.get_transform().location
    spawn_points = [
        point for point in world.get_map().get_spawn_points()
        if point.location.distance(ego_location) >= 18.0
    ]
    random.shuffle(spawn_points)

    # Spawn rare vehicle classes deliberately.
    rare_blueprints = [
        *sorted(MOTORCYCLES),
        *sorted(BICYCLES),
        *sorted(TRUCKS),
        *sorted(BUSES),
    ]

    available_ids = {
        bp.id for bp in blueprint_library.filter("vehicle.*")
    }

    rare_blueprints = [
        name for name in rare_blueprints
        if name in available_ids
    ]

    ordinary_blueprints = [
        bp for bp in blueprint_library.filter("vehicle.*")
        if bp.has_attribute("number_of_wheels")
        and bp.id not in MOTORCYCLES
        and bp.id not in BICYCLES
        and bp.id not in TRUCKS
        and bp.id not in BUSES
    ]

    rare_bps = [
        blueprint_library.find(name)
        for name in rare_blueprints
    ]

    spawned = []
    spawn_log = Counter()

    # Retry each rare blueprint at different locations. The previous
    # index-to-spawn-point pairing silently lost a class when its one
    # selected location was occupied.
    remaining_points = list(spawn_points)
    random.shuffle(remaining_points)

    def try_spawn_blueprint(blueprint, consume_failed_points=False):
        for spawn_point in list(remaining_points):
            try:
                actor = world.try_spawn_actor(blueprint, spawn_point)
            except RuntimeError:
                actor = None

            if actor is None:
                if consume_failed_points:
                    remaining_points.remove(spawn_point)
                continue

            # A successful spawn occupies this point. Failed rare-blueprint
            # attempts remain available for other rare vehicle types.
            remaining_points.remove(spawn_point)

            try:
                if blueprint.id in MOTORCYCLES | BICYCLES | BUSES:
                    # Keep target classes at known positions for the
                    # supplemental close-up views captured after the drive.
                    actor.set_autopilot(False)
                else:
                    actor.set_autopilot(True, traffic_manager.get_port())
            except RuntimeError:
                actor.destroy()
                continue

            spawned.append(actor)
            spawn_log[blueprint.id] += 1
            return True
        return False

    for blueprint in rare_bps:
        if len(spawned) >= NUM_TRAFFIC_VEHICLES or not remaining_points:
            break
        if not try_spawn_blueprint(blueprint):
            print(f"WARNING: Could not spawn rare vehicle {blueprint.id}")

    while (
        len(spawned) < NUM_TRAFFIC_VEHICLES
        and remaining_points
        and ordinary_blueprints
    ):
        blueprint = random.choice(ordinary_blueprints)
        try_spawn_blueprint(blueprint, consume_failed_points=True)

    print(f"\nTraffic vehicles spawned: {len(spawned)}")

    for vehicle_id, count in sorted(spawn_log.items()):
        print(f"  {vehicle_id}: {count}")

    spawned_types = set(spawn_log)
    for class_name, class_blueprints in (
        ("bicycle", BICYCLES),
        ("bus", BUSES),
    ):
        if not spawned_types.intersection(class_blueprints):
            print(f"WARNING: No {class_name} spawned; its class will be absent.")

    return spawned


def spawn_pedestrians(world):
    blueprint_library = world.get_blueprint_library()

    walker_blueprints = blueprint_library.filter(
        "walker.pedestrian.*"
    )

    controller_blueprint = blueprint_library.find(
        "controller.ai.walker"
    )

    walkers = []
    controllers = []

    for _ in range(NUM_WALKERS):
        location = world.get_random_location_from_navigation()

        if location is None:
            continue

        walker_bp = random.choice(walker_blueprints)

        try:
            walker = world.try_spawn_actor(
                walker_bp,
                carla.Transform(location),
            )

            if walker is None:
                continue

            controller = world.spawn_actor(
                controller_blueprint,
                carla.Transform(),
                attach_to=walker,
            )

            walkers.append(walker)
            controllers.append(controller)

        except RuntimeError:
            continue

    # Start controllers after all successful spawns.
    world.tick()

    for controller in controllers:
        try:
            controller.start()

            destination = world.get_random_location_from_navigation()

            if destination is not None:
                controller.go_to_location(destination)

            controller.set_max_speed(random.uniform(1.0, 1.8))

        except RuntimeError:
            pass

    print(f"Pedestrians spawned: {len(walkers)}")

    return walkers, controllers


# ============================================================
# DATA COLLECTION
# ============================================================

def main():
    client = carla.Client(HOST, PORT)
    client.set_timeout(20.0)

    world = client.get_world()
    traffic_manager = client.get_trafficmanager()

    original_settings = world.get_settings()
    original_tm_sync = None

    ego_vehicle = None
    camera = None
    semantic_camera = None
    target_camera = None
    target_semantic_camera = None

    spawned_vehicles = []
    spawned_walkers = []
    spawned_controllers = []

    image_queue = queue.Queue()
    semantic_queue = queue.Queue()
    target_image_queue = queue.Queue()
    target_semantic_queue = queue.Queue()

    class_counts = Counter()
    image_counts = Counter()
    empty_images = 0
    skipped_frames = 0
    saved_count = 0

    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = FIXED_DELTA_SECONDS
        world.apply_settings(settings)

        # Preserve the Traffic Manager's original sync state
        # when the API supports reading it.
        try:
            original_tm_sync = traffic_manager.get_synchronous_mode()
        except (AttributeError, RuntimeError):
            original_tm_sync = None

        traffic_manager.set_synchronous_mode(True)
        traffic_manager.set_random_device_seed(RANDOM_SEED)

        blueprint_library = world.get_blueprint_library()

        # Spawn the ego vehicle.
        ego_bp = blueprint_library.find("vehicle.tesla.model3")
        spawn_points = world.get_map().get_spawn_points()
        random.shuffle(spawn_points)

        for spawn_point in spawn_points:
            ego_vehicle = world.try_spawn_actor(
                ego_bp,
                spawn_point,
            )

            if ego_vehicle is not None:
                break

        if ego_vehicle is None:
            raise RuntimeError("Could not spawn the ego vehicle.")

        ego_vehicle.set_autopilot(
            True,
            traffic_manager.get_port(),
        )

        print("Ego vehicle:", ego_vehicle.type_id)

        spawned_vehicles = spawn_traffic(
            world,
            traffic_manager,
            ego_vehicle,
        )

        spawned_walkers, spawned_controllers = spawn_pedestrians(
            world,
        )

        # Configure the camera.
        camera_bp = blueprint_library.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(IMAGE_WIDTH))
        camera_bp.set_attribute("image_size_y", str(IMAGE_HEIGHT))
        camera_bp.set_attribute("fov", str(CAMERA_FOV))
        camera_bp.set_attribute("sensor_tick", str(FIXED_DELTA_SECONDS))

        camera_mount = carla.Transform(
            carla.Location(x=1.6, z=1.7)
        )

        camera = world.spawn_actor(
            camera_bp,
            camera_mount,
            attach_to=ego_vehicle,
        )

        camera.listen(image_queue.put)

        semantic_bp = blueprint_library.find(
            "sensor.camera.semantic_segmentation"
        )
        semantic_bp.set_attribute("image_size_x", str(IMAGE_WIDTH))
        semantic_bp.set_attribute("image_size_y", str(IMAGE_HEIGHT))
        semantic_bp.set_attribute("fov", str(CAMERA_FOV))
        semantic_bp.set_attribute("sensor_tick", str(FIXED_DELTA_SECONDS))
        semantic_camera = world.spawn_actor(
            semantic_bp,
            camera_mount,
            attach_to=ego_vehicle,
        )
        semantic_camera.listen(semantic_queue.put)

        camera_matrix = build_camera_matrix(
            IMAGE_WIDTH,
            IMAGE_HEIGHT,
            CAMERA_FOV,
        )

        traffic_sign_boxes = get_traffic_sign_bounding_boxes(world)
        print(f"Map traffic-sign bounding boxes available: {len(traffic_sign_boxes)}")

        target_camera_bp = blueprint_library.find("sensor.camera.rgb")
        target_camera_bp.set_attribute("image_size_x", str(IMAGE_WIDTH))
        target_camera_bp.set_attribute("image_size_y", str(IMAGE_HEIGHT))
        target_camera_bp.set_attribute("fov", str(CAMERA_FOV))
        target_camera_bp.set_attribute("sensor_tick", str(FIXED_DELTA_SECONDS))

        target_semantic_bp = blueprint_library.find(
            "sensor.camera.semantic_segmentation"
        )
        target_semantic_bp.set_attribute("image_size_x", str(IMAGE_WIDTH))
        target_semantic_bp.set_attribute("image_size_y", str(IMAGE_HEIGHT))
        target_semantic_bp.set_attribute("fov", str(CAMERA_FOV))
        target_semantic_bp.set_attribute("sensor_tick", str(FIXED_DELTA_SECONDS))

        existing_indices = [
            int(match.group(1))
            for path in IMAGE_DIR.glob("pilot_*.png")
            if (match := re.fullmatch(r"pilot_(\d+)", path.stem))
        ]
        filename_offset = max(existing_indices, default=-1) + 1

        print("\nWarming up simulation...")

        for _ in range(30):
            world.tick()

        # Drain images produced while sensors were warming up.
        for sensor_queue in (image_queue, semantic_queue):
            while True:
                try:
                    sensor_queue.get_nowait()
                except queue.Empty:
                    break

        print(f"\nCollecting {NUM_IMAGES} images...")

        for index in range(NUM_IMAGES):
            # Let traffic move between saved images. Advance through the
            # whole interval, then pair the final world frame with its image.
            for _ in range(CAPTURE_INTERVAL_FRAMES - 1):
                world.tick()

            frame_id = world.tick()

            matched_image = wait_for_sensor_frame(
                image_queue, frame_id, "RGB camera"
            )
            semantic_image = wait_for_sensor_frame(
                semantic_queue, frame_id, "semantic camera"
            )
            if matched_image is None or semantic_image is None:
                skipped_frames += 1
                continue

            # Get world snapshot associated with this frame.
            snapshot = world.get_snapshot()

            # Guard against any frame mismatch.
            if snapshot.frame != frame_id:
                skipped_frames += 1
                print(f"Skipped frame {frame_id}: snapshot mismatch.")
                continue

            sign_mask = get_semantic_tag_mask(semantic_image, tag_id=8)
            labels, frame_class_counts = build_frame_labels(
                world,
                snapshot,
                camera,
                camera_matrix,
                ego_vehicle,
                traffic_sign_boxes,
                sign_mask,
            )

            # Keep previous collection runs instead of overwriting them.
            filename = f"pilot_{filename_offset + saved_count:05d}"

            try:
                was_empty = save_frame(
                    matched_image,
                    labels,
                    filename,
                    frame_class_counts,
                    class_counts,
                )
            except RuntimeError as exc:
                skipped_frames += 1
                print(exc)
                continue

            saved_count += 1

            if was_empty:
                empty_images += 1

            image_counts.update(frame_class_counts.keys())

            if saved_count % 5 == 0:
                print(
                    f"Saved {saved_count}/{NUM_IMAGES} "
                    f"| boxes in last image: {len(labels)}"
                )

        # Add controlled, unobstructed viewpoints for rare spawned vehicles.
        # This guarantees the pilot contains visible examples when CARLA
        # successfully spawned those blueprints, even if the ego route misses them.
        target_view_offsets = (-20, -10, 0, 10, 20)
        target_distances = (12, 14, 16, 14, 12)
        targeted_image_count = 0

        for actor in spawned_vehicles:
            target_class = get_class_id(actor)
            if target_class not in {2, 3, 7}:
                continue

            actor_state = world.get_snapshot().find(actor.id)
            if actor_state is None:
                continue
            target_transform = actor_state.get_transform()
            target_location = target_transform.location

            for view_index in range(TARGETED_VIEWS_PER_TARGET_ACTOR):
                orbit_angle = math.radians(
                    target_transform.rotation.yaw
                    + 180
                    + target_view_offsets[view_index]
                )
                camera_location = carla.Location(
                    x=target_location.x + target_distances[view_index] * math.cos(orbit_angle),
                    y=target_location.y + target_distances[view_index] * math.sin(orbit_angle),
                    z=target_location.z + 2.2,
                )
                dx = target_location.x - camera_location.x
                dy = target_location.y - camera_location.y
                dz = target_location.z + 0.7 - camera_location.z
                camera_rotation = carla.Rotation(
                    yaw=math.degrees(math.atan2(dy, dx)),
                    pitch=math.degrees(math.atan2(dz, math.hypot(dx, dy))),
                )
                target_pose = carla.Transform(
                    camera_location,
                    camera_rotation,
                )
                if target_camera is None:
                    target_camera = world.spawn_actor(
                        target_camera_bp,
                        target_pose,
                    )
                    target_camera.listen(target_image_queue.put)
                    target_semantic_camera = world.spawn_actor(
                        target_semantic_bp,
                        target_pose,
                    )
                    target_semantic_camera.listen(target_semantic_queue.put)

                    # Let both sensors register and emit startup frames before
                    # relying on paired frames. The first measurement can lag
                    # sensor creation by more than one synchronous tick.
                    for _ in range(3):
                        world.tick()
                    for sensor_queue in (
                        target_image_queue,
                        target_semantic_queue,
                    ):
                        while True:
                            try:
                                sensor_queue.get_nowait()
                            except queue.Empty:
                                break
                else:
                    target_camera.set_transform(target_pose)
                    target_semantic_camera.set_transform(target_pose)

                frame_id = world.tick()
                rgb_image = wait_for_sensor_frame(
                    target_image_queue, frame_id, "target RGB camera"
                )
                semantic_image = wait_for_sensor_frame(
                    target_semantic_queue, frame_id, "target semantic camera"
                )
                if rgb_image is None or semantic_image is None:
                    skipped_frames += 1
                    continue

                snapshot = world.get_snapshot()
                if snapshot.frame != frame_id:
                    skipped_frames += 1
                    print(f"Skipped target frame {frame_id}: snapshot mismatch.")
                    continue

                sign_mask = get_semantic_tag_mask(semantic_image, tag_id=8)
                labels, frame_class_counts = build_frame_labels(
                    world,
                    snapshot,
                    target_camera,
                    camera_matrix,
                    ego_vehicle,
                    traffic_sign_boxes,
                    sign_mask,
                )
                filename = f"pilot_{filename_offset + saved_count:05d}"
                try:
                    was_empty = save_frame(
                        rgb_image,
                        labels,
                        filename,
                        frame_class_counts,
                        class_counts,
                    )
                except RuntimeError as exc:
                    skipped_frames += 1
                    print(exc)
                    continue

                saved_count += 1
                targeted_image_count += 1
                if was_empty:
                    empty_images += 1
                image_counts.update(frame_class_counts.keys())

        print(f"Supplemental rare-vehicle views saved: {targeted_image_count}")

        # Create a summary report.
        report_path = OUTPUT_DIR / "dataset_report.txt"

        with open(report_path, "w", encoding="utf-8") as report:
            report.write("CARLA BALANCED DATASET PILOT REPORT\n")
            report.write("=" * 45 + "\n")
            report.write(f"CARLA version: {client.get_server_version()}\n")
            report.write(f"Map: {world.get_map().name}\n")
            report.write(f"Random seed: {RANDOM_SEED}\n")
            report.write(f"Requested images: {NUM_IMAGES}\n")
            report.write(
                f"Targeted views per spawned motorcycle/bicycle/bus actor: "
                f"{TARGETED_VIEWS_PER_TARGET_ACTOR}\n"
            )
            report.write(f"Supplemental rare-vehicle images saved: {targeted_image_count}\n")
            report.write(
                f"Capture interval (simulation frames): "
                f"{CAPTURE_INTERVAL_FRAMES}\n"
            )
            report.write(f"Saved images: {saved_count}\n")
            report.write(f"Skipped frames: {skipped_frames}\n")
            report.write(f"Images with empty labels: {empty_images}\n")
            report.write(f"Total boxes: {sum(class_counts.values())}\n\n")
            report.write("OBJECT COUNTS BY CLASS\n")

            for class_id, class_name in enumerate(CLASS_NAMES):
                report.write(
                    f"{class_id} - {class_name}: "
                    f"{class_counts[class_id]}\n"
                )

            report.write("\nIMAGES CONTAINING EACH CLASS\n")

            for class_id, class_name in enumerate(CLASS_NAMES):
                report.write(
                    f"{class_id} - {class_name}: "
                    f"{image_counts[class_id]}\n"
                )

        print("\n" + "=" * 55)
        print("PILOT COLLECTION COMPLETE")
        print("=" * 55)
        print("Images saved:", saved_count)
        print("Frames skipped:", skipped_frames)
        print("Images with empty labels:", empty_images)
        print("Total bounding boxes:", sum(class_counts.values()))

        print("\nOBJECT COUNTS BY CLASS")

        for class_id, class_name in enumerate(CLASS_NAMES):
            print(
                f"{class_id} - {class_name}: "
                f"{class_counts[class_id]} boxes "
                f"in {image_counts[class_id]} images"
            )

        print("\nImage directory:", IMAGE_DIR)
        print("Label directory:", LABEL_DIR)
        print("Report:", report_path)

    finally:
        print("\nCleaning up...")

        if camera is not None:
            try:
                camera.stop()
            except RuntimeError:
                pass
            try:
                camera.destroy()
            except RuntimeError:
                pass

        for sensor in (
            semantic_camera,
            target_camera,
            target_semantic_camera,
        ):
            if sensor is None:
                continue
            try:
                sensor.stop()
            except RuntimeError:
                pass
            try:
                sensor.destroy()
            except RuntimeError:
                pass

        for controller in spawned_controllers:
            try:
                controller.stop()
            except RuntimeError:
                pass

        for actor in (
            spawned_controllers
            + spawned_walkers
            + spawned_vehicles
        ):
            try:
                if actor.is_alive:
                    actor.destroy()
            except RuntimeError:
                pass

        if ego_vehicle is not None:
            try:
                if ego_vehicle.is_alive:
                    ego_vehicle.destroy()
            except RuntimeError:
                pass

        try:
            if original_tm_sync is not None:
                traffic_manager.set_synchronous_mode(original_tm_sync)
            else:
                traffic_manager.set_synchronous_mode(False)
        except RuntimeError:
            pass

        try:
            world.apply_settings(original_settings)
        except RuntimeError:
            pass

        print("Cleanup complete.")


if __name__ == "__main__":
    main()
