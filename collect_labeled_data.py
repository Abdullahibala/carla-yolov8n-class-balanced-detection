import carla
import cv2
import numpy as np
import queue
import random
import time
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

HOST = "127.0.0.1"
PORT = 2000

NUM_IMAGES = 300
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720

FPS = 20
FIXED_DELTA_SECONDS = 1.0 / FPS

NUM_TRAFFIC_VEHICLES = 30
NUM_WALKERS = 10

MAX_OBJECT_DISTANCE = 60.0
MIN_BOX_AREA = 25.0

OUTPUT_DIR = (
    Path(__file__).resolve().parent
    / "carla_dataset"
    / "labeled_dataset"
)

IMAGE_DIR = OUTPUT_DIR / "images"
LABEL_DIR = OUTPUT_DIR / "labels"

# YOLO class IDs
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


# ============================================================
# DIRECTORIES
# ============================================================

IMAGE_DIR.mkdir(parents=True, exist_ok=True)
LABEL_DIR.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_DIR / "classes.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(CLASS_NAMES))


# ============================================================
# CAMERA AND PROJECTION HELPERS
# ============================================================

def build_camera_matrix(width, height, fov_degrees):
    """Construct the camera intrinsic matrix."""

    focal_length = width / (
        2.0 * np.tan(np.radians(fov_degrees) / 2.0)
    )

    return np.array(
        [
            [focal_length, 0.0, width / 2.0],
            [0.0, focal_length, height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def get_class_id(actor):
    """
    Map CARLA actor types to our YOLO classes.

    Classes:
        0: car
        1: pedestrian
        2: motorcycle
        3: bicycle
        4: traffic_light
        5: traffic_sign
        6: truck
        7: bus
    """

    type_id = actor.type_id.lower()

    # --------------------------------------------------------
    # PEDESTRIANS
    # --------------------------------------------------------
    if type_id.startswith("walker.pedestrian"):
        return 1

    # --------------------------------------------------------
    # TRAFFIC LIGHTS
    # --------------------------------------------------------
    if type_id == "traffic.traffic_light":
        return 4

    # --------------------------------------------------------
    # TRAFFIC SIGNS
    # --------------------------------------------------------
    if type_id in {
        "traffic.stop",
        "traffic.yield",
    }:
        return 5

    # --------------------------------------------------------
    # VEHICLES
    # --------------------------------------------------------
    if type_id.startswith("vehicle."):

        # Explicit motorcycle blueprints
        motorcycle_names = {
            "vehicle.harley-davidson.low_rider",
            "vehicle.kawasaki.ninja",
            "vehicle.vespa.zx125",
            "vehicle.yamaha.yzf",
        }

        if type_id in motorcycle_names:
            return 2

        # Explicit bicycle blueprints
        bicycle_names = {
            "vehicle.bh.crossbike",
            "vehicle.diamondback.century",
            "vehicle.gazelle.omafiets",
        }

        if type_id in bicycle_names:
            return 3

        # Explicit truck / heavy-vehicle blueprints
        truck_names = {
            "vehicle.carlamotors.carlacola",
            "vehicle.carlamotors.european_hgv",
            "vehicle.carlamotors.firetruck",
            "vehicle.tesla.cybertruck",
        }

        if type_id in truck_names:
            return 6

        # Explicit bus blueprints, if the installed CARLA
        # version provides any.
        bus_names = {
            "vehicle.mitsubishi.fusorosa",
        }

        if type_id in bus_names:
            return 7

        # Remaining supported vehicles are assigned to car.
        return 0

    # Unrecognized actors are ignored.
    return None


def world_to_camera(point, camera_inverse_matrix):
    """Transform a world-space point into camera coordinates."""

    point_h = np.array(
        [point.x, point.y, point.z, 1.0],
        dtype=np.float64,
    )

    local = camera_inverse_matrix @ point_h

    # CARLA camera-local axes:
    # X = forward, Y = right, Z = up
    #
    # Pinhole camera axes:
    # X = right, Y = down, Z = forward

    return np.array(
        [local[1], -local[2], local[0]],
        dtype=np.float64,
    )


def project_actor_bbox(actor, camera_transform, camera_matrix):
    """
    Project a CARLA actor's 3D bounding box into the image.

    Returns a clipped 2D box:
        x_min, y_min, x_max, y_max

    Returns None if the box cannot be projected reliably.
    """

    try:
        world_vertices = actor.bounding_box.get_world_vertices(
            actor.get_transform()
        )
    except (RuntimeError, AttributeError):
        return None

    camera_inverse = np.array(
        camera_transform.get_inverse_matrix(),
        dtype=np.float64,
    )

    projected_points = []

    for vertex in world_vertices:
        camera_point = world_to_camera(
            vertex,
            camera_inverse,
        )

        x_cam, y_cam, z_cam = camera_point

        # Reject boxes crossing the camera plane.
        # This avoids unstable projections near or behind the camera.
        if z_cam <= 0.1:
            return None

        pixel = camera_matrix @ camera_point

        u = pixel[0] / pixel[2]
        v = pixel[1] / pixel[2]

        projected_points.append([u, v])

    projected_points = np.asarray(
        projected_points,
        dtype=np.float64,
    )

    x_min = projected_points[:, 0].min()
    y_min = projected_points[:, 1].min()
    x_max = projected_points[:, 0].max()
    y_max = projected_points[:, 1].max()

    # Reject boxes entirely outside the image.
    if (
        x_max <= 0
        or y_max <= 0
        or x_min >= IMAGE_WIDTH
        or y_min >= IMAGE_HEIGHT
    ):
        return None

    # Clip the box to image boundaries.
    x_min = np.clip(x_min, 0, IMAGE_WIDTH - 1)
    y_min = np.clip(y_min, 0, IMAGE_HEIGHT - 1)
    x_max = np.clip(x_max, 0, IMAGE_WIDTH - 1)
    y_max = np.clip(y_max, 0, IMAGE_HEIGHT - 1)

    box_width = x_max - x_min
    box_height = y_max - y_min

    if box_width <= 0 or box_height <= 0:
        return None

    if box_width * box_height < MIN_BOX_AREA:
        return None

    return x_min, y_min, x_max, y_max


def to_yolo_format(class_id, box):
    """Convert pixel coordinates into normalized YOLO coordinates."""

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
# TRAFFIC AND PEDESTRIAN SPAWNING
# ============================================================

def spawn_traffic(client, world, traffic_manager, ego_vehicle):
    """Spawn additional vehicles and enable their autopilot."""

    blueprint_library = world.get_blueprint_library()

    vehicle_blueprints = blueprint_library.filter("vehicle.*")

    # Avoid unsuitable or unusual vehicle types where possible.
    vehicle_blueprints = [
        bp for bp in vehicle_blueprints
        if bp.has_attribute("number_of_wheels")
    ]

    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)

    spawned = []

    for spawn_point in spawn_points:
        if len(spawned) >= NUM_TRAFFIC_VEHICLES:
            break

        if random.random() < 0.15:
            continue

        blueprint = random.choice(vehicle_blueprints)

        try:
            actor = world.try_spawn_actor(
                blueprint,
                spawn_point,
            )

            if actor is None:
                continue

            if actor.id == ego_vehicle.id:
                actor.destroy()
                continue

            actor.set_autopilot(
                True,
                traffic_manager.get_port(),
            )

            spawned.append(actor)

        except RuntimeError:
            continue

    print(f"Spawned {len(spawned)} traffic vehicles.")

    return spawned


def spawn_pedestrians(client, world):
    """Spawn pedestrians and their AI controllers."""

    blueprint_library = world.get_blueprint_library()

    walker_blueprints = blueprint_library.filter(
        "walker.pedestrian.*"
    )

    controller_blueprint = blueprint_library.find(
        "controller.ai.walker"
    )

    spawned_walkers = []
    spawned_controllers = []

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

            spawned_walkers.append(walker)
            spawned_controllers.append(controller)

        except RuntimeError:
            continue

    # Controllers should be started after spawning has finished.
    world.tick()

    for controller in spawned_controllers:
        try:
            controller.start()

            destination = world.get_random_location_from_navigation()

            if destination is not None:
                controller.go_to_location(destination)

            controller.set_max_speed(random.uniform(1.0, 1.8))

        except RuntimeError:
            pass

    print(f"Spawned {len(spawned_walkers)} pedestrians.")

    return spawned_walkers, spawned_controllers


# ============================================================
# MAIN COLLECTION PIPELINE
# ============================================================

def main():
    client = carla.Client(HOST, PORT)
    client.set_timeout(20.0)

    world = client.get_world()
    traffic_manager = client.get_trafficmanager()

    original_settings = world.get_settings()

    ego_vehicle = None
    camera = None

    spawned_vehicles = []
    spawned_walkers = []
    spawned_controllers = []

    image_queue = queue.Queue()

    previous_settings = None

    try:
        # Configure synchronous simulation.
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = FIXED_DELTA_SECONDS

        world.apply_settings(settings)

        traffic_manager.set_synchronous_mode(True)

        # Use a repeatable random seed for initial reproducibility.
        random.seed(42)

        blueprint_library = world.get_blueprint_library()

        # Spawn ego vehicle.
        vehicle_bp = blueprint_library.find(
            "vehicle.tesla.model3"
        )

        spawn_points = world.get_map().get_spawn_points()
        random.shuffle(spawn_points)

        for spawn_point in spawn_points:
            ego_vehicle = world.try_spawn_actor(
                vehicle_bp,
                spawn_point,
            )

            if ego_vehicle is not None:
                break

        if ego_vehicle is None:
            raise RuntimeError(
                "Could not spawn ego vehicle. "
                "Please check the available spawn points."
            )

        print("Ego vehicle spawned:", ego_vehicle.type_id)

        ego_vehicle.set_autopilot(
            True,
            traffic_manager.get_port(),
        )

        traffic_manager.set_random_device_seed(42)

        # Spawn surrounding traffic.
        spawned_vehicles = spawn_traffic(
            client,
            world,
            traffic_manager,
            ego_vehicle,
        )

        # Spawn pedestrians.
        spawned_walkers, spawned_controllers = spawn_pedestrians(
            client,
            world,
        )

        # Attach RGB camera to the ego vehicle.
        camera_bp = blueprint_library.find(
            "sensor.camera.rgb"
        )

        camera_bp.set_attribute(
            "image_size_x",
            str(IMAGE_WIDTH),
        )
        camera_bp.set_attribute(
            "image_size_y",
            str(IMAGE_HEIGHT),
        )
        camera_bp.set_attribute(
            "fov",
            "90",
        )
        camera_bp.set_attribute(
            "sensor_tick",
            str(FIXED_DELTA_SECONDS),
        )

        camera_transform = carla.Transform(
            carla.Location(x=1.6, z=1.7)
        )

        camera = world.spawn_actor(
            camera_bp,
            camera_transform,
            attach_to=ego_vehicle,
        )

        camera.listen(
            lambda image: image_queue.put(image)
        )

        camera_matrix = build_camera_matrix(
            IMAGE_WIDTH,
            IMAGE_HEIGHT,
            90.0,
        )

        print("\nWarming up simulation...")

        # Advance simulation so camera and traffic settle.
        for _ in range(20):
            world.tick()

        # Discard old sensor frames before collecting.
        while not image_queue.empty():
            try:
                image_queue.get_nowait()
            except queue.Empty:
                break

        print(f"\nCollecting {NUM_IMAGES} labeled images...")

        saved_count = 0
        skipped_count = 0

        # Track actor IDs so the ego vehicle is never labeled.
        for index in range(NUM_IMAGES):
            frame_id = world.tick()

            # Find the camera image belonging to this world frame.
            matched_image = None
            deadline = time.monotonic() + 10.0

            while time.monotonic() < deadline:
                try:
                    image = image_queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                if image.frame == frame_id:
                    matched_image = image
                    break

                if image.frame > frame_id:
                    # We missed the requested frame.
                    break

            if matched_image is None:
                skipped_count += 1
                print(
                    f"Skipping frame {frame_id}: "
                    "matching camera frame unavailable."
                )
                continue

            # Convert BGRA sensor buffer to BGR for OpenCV.
            image_array = np.frombuffer(
                matched_image.raw_data,
                dtype=np.uint8,
            )

            image_array = image_array.reshape(
                IMAGE_HEIGHT,
                IMAGE_WIDTH,
                4,
            )

            image_bgr = image_array[:, :, :3].copy()

            # Get the camera pose for this captured frame.
            camera_pose = camera.get_transform()

            # Collect actors and generate annotations.
            labels = []

            actors = world.get_actors()

            for actor in actors:
                if not actor.is_alive:
                    continue

                if actor.id == ego_vehicle.id:
                    continue

                class_id = get_class_id(actor)

                if class_id is None:
                    continue

                # Exclude distant objects.
                distance = actor.get_location().distance(
                    camera_pose.location
                )

                if distance > MAX_OBJECT_DISTANCE:
                    continue

                box = project_actor_bbox(
                    actor,
                    camera_pose,
                    camera_matrix,
                )

                if box is None:
                    continue

                labels.append(
                    to_yolo_format(class_id, box)
                )

            # Save image and label with matching filename.
            filename = f"driving_{saved_count:05d}"

            image_path = IMAGE_DIR / f"{filename}.png"
            label_path = LABEL_DIR / f"{filename}.txt"

            success = cv2.imwrite(
                str(image_path),
                image_bgr,
            )

            if not success:
                print("Failed to save:", image_path)
                skipped_count += 1
                continue

            with open(
                label_path,
                "w",
                encoding="utf-8",
            ) as label_file:
                label_file.write("\n".join(labels))

            saved_count += 1

            if saved_count % 10 == 0:
                print(
                    f"Saved {saved_count}/{NUM_IMAGES} "
                    f"| Objects in last image: {len(labels)}"
                )

        print("\nCollection complete.")
        print("Images saved:", saved_count)
        print("Frames skipped:", skipped_count)
        print("Image directory:", IMAGE_DIR)
        print("Label directory:", LABEL_DIR)

    finally:
        print("\nCleaning up CARLA actors...")

        if camera is not None:
            try:
                camera.stop()
            except RuntimeError:
                pass

            try:
                camera.destroy()
            except RuntimeError:
                pass

        for controller in spawned_controllers:
            try:
                controller.stop()
            except RuntimeError:
                pass

        actors_to_destroy = (
            spawned_controllers
            + spawned_walkers
            + spawned_vehicles
        )

        for actor in actors_to_destroy:
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
            traffic_manager.set_synchronous_mode(False)
        except RuntimeError:
            pass

        # Restore the simulator settings that existed before collection.
        world.apply_settings(original_settings)

        print("Cleanup complete.")


if __name__ == "__main__":
    main()