
import carla
import os
import random
import queue
import pathlib
import time
import numpy as np
import cv2

# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_DIR = pathlib.Path(
    "carla_dataset/driving_images"
).resolve()

NUM_IMAGES = 300

IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720

FIXED_DELTA_SECONDS = 0.05
CAPTURE_INTERVAL = 0.5

NUM_TRAFFIC_VEHICLES = 30
NUM_PEDESTRIANS = 10

CARLA_HOST = "localhost"
CARLA_PORT = 2000

IMAGE_TIMEOUT = 10.0

RANDOM_SEED = 42


# ============================================================
# IMAGE CALLBACK
# ============================================================

def camera_callback(image_queue):
    def callback(image):
        image_queue.put((image.frame, image))
    return callback


# ============================================================
# IMAGE SAVING
# ============================================================

def save_camera_image(image, output_dir, index):
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = output_dir / f"driving_{index:05d}.png"

    image_array = np.frombuffer(
        image.raw_data,
        dtype=np.uint8
    ).reshape(
        image.height,
        image.width,
        4
    )

    success = cv2.imwrite(
        str(filename),
        image_array
    )

    if not success:
        raise RuntimeError(
            f"Failed to save image: {filename}"
        )

    return filename


# ============================================================
# MAIN
# ============================================================

def main():

    random.seed(RANDOM_SEED)

    client = carla.Client(CARLA_HOST, CARLA_PORT)
    client.set_timeout(30.0)

    world = client.get_world()
    blueprint_library = world.get_blueprint_library()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Remove images from the previous run so results are clear.
    for old_file in OUTPUT_DIR.glob("driving_*.png"):
        old_file.unlink()

    vehicle = None
    camera = None

    traffic_vehicles = []
    pedestrians = []
    walker_controllers = []

    image_queue = queue.Queue()

    original_settings = world.get_settings()
    traffic_manager = client.get_trafficmanager()

    # Preserve existing traffic manager settings where available.
    original_tm_sync = False

    saved = 0

    try:
        # ----------------------------------------------------
        # SYNCHRONOUS SIMULATION
        # ----------------------------------------------------

        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = FIXED_DELTA_SECONDS

        world.apply_settings(settings)

        traffic_manager.set_synchronous_mode(True)
        original_tm_sync = True

        print("Synchronous simulation enabled.")

        # ----------------------------------------------------
        # SPAWN EGO VEHICLE
        # ----------------------------------------------------

        spawn_points = world.get_map().get_spawn_points()
        random.shuffle(spawn_points)

        vehicle_bp = blueprint_library.find(
            "vehicle.tesla.model3"
        )

        for spawn_point in spawn_points:
            vehicle = world.try_spawn_actor(
                vehicle_bp,
                spawn_point
            )
            if vehicle is not None:
                break

        if vehicle is None:
            raise RuntimeError(
                "Could not spawn the ego vehicle."
            )

        vehicle.set_autopilot(
            True,
            traffic_manager.get_port()
        )

        print("Ego vehicle spawned:", vehicle.type_id)

        # ----------------------------------------------------
        # SPAWN TRAFFIC VEHICLES
        # ----------------------------------------------------

        vehicle_blueprints = blueprint_library.filter(
            "vehicle.*"
        )

        random.shuffle(spawn_points)

        for spawn_point in spawn_points:
            if len(traffic_vehicles) >= NUM_TRAFFIC_VEHICLES:
                break

            blueprint = random.choice(vehicle_blueprints)

            actor = world.try_spawn_actor(
                blueprint,
                spawn_point
            )

            if actor is not None:
                if actor.id != vehicle.id:
                    actor.set_autopilot(
                        True,
                        traffic_manager.get_port()
                    )
                    traffic_vehicles.append(actor)

        print(
            f"Traffic vehicles spawned: "
            f"{len(traffic_vehicles)}"
        )

        # ----------------------------------------------------
        # SPAWN PEDESTRIANS
        # ----------------------------------------------------

        walker_blueprints = blueprint_library.filter(
            "walker.pedestrian.*"
        )

        controller_bp = blueprint_library.find(
            "controller.ai.walker"
        )

        for _ in range(NUM_PEDESTRIANS):
            location = world.get_random_location_from_navigation()

            if location is None:
                continue

            walker_bp = random.choice(walker_blueprints)

            walker = world.try_spawn_actor(
                walker_bp,
                carla.Transform(location)
            )

            if walker is not None:
                pedestrians.append(walker)

        # Advance one frame before spawning walker controllers.
        world.tick()

        for walker in pedestrians:
            controller = world.spawn_actor(
                controller_bp,
                carla.Transform(),
                attach_to=walker
            )

            walker_controllers.append(controller)

            controller.start()

            destination = world.get_random_location_from_navigation()

            if destination is not None:
                controller.go_to_location(destination)

            controller.set_max_speed(
                random.uniform(0.8, 1.8)
            )

        print(
            f"Pedestrians spawned: {len(pedestrians)}"
        )

        # ----------------------------------------------------
        # CONFIGURE CAMERA
        # ----------------------------------------------------

        camera_bp = blueprint_library.find(
            "sensor.camera.rgb"
        )

        camera_bp.set_attribute(
            "image_size_x",
            str(IMAGE_WIDTH)
        )

        camera_bp.set_attribute(
            "image_size_y",
            str(IMAGE_HEIGHT)
        )

        camera_bp.set_attribute("fov", "90")

        camera_bp.set_attribute(
            "sensor_tick",
            str(FIXED_DELTA_SECONDS)
        )

        camera_transform = carla.Transform(
            carla.Location(x=1.5, z=1.6)
        )

        camera = world.spawn_actor(
            camera_bp,
            camera_transform,
            attach_to=vehicle
        )

        camera.listen(camera_callback(image_queue))

        print("Camera spawned.")

        # ----------------------------------------------------
        # WARM-UP
        # ----------------------------------------------------

        print("Warming up the simulation...")

        for _ in range(40):
            world.tick()

        # Discard warm-up images.
        while True:
            try:
                image_queue.get_nowait()
            except queue.Empty:
                break

        print("Starting diverse image collection...")

        # ----------------------------------------------------
        # COLLECT SYNCHRONIZED IMAGES
        # ----------------------------------------------------

        capture_every_n_frames = max(
            1,
            round(CAPTURE_INTERVAL / FIXED_DELTA_SECONDS)
        )

        last_capture_frame = -capture_every_n_frames

        while saved < NUM_IMAGES:

            frame_id = world.tick()

            if (
                frame_id - last_capture_frame
                < capture_every_n_frames
            ):
                continue

            deadline = time.monotonic() + IMAGE_TIMEOUT
            matching_image = None

            while time.monotonic() < deadline:
                try:
                    image_frame, image = image_queue.get(
                        timeout=0.2
                    )
                except queue.Empty:
                    continue

                if image_frame == frame_id:
                    matching_image = image
                    break

                # Ignore stale images.
                if image_frame > frame_id:
                    print(
                        "Warning: camera frame is ahead of "
                        "the world frame."
                    )

            if matching_image is None:
                print(
                    f"ERROR: No image for world frame "
                    f"{frame_id}."
                )
                break

            filename = save_camera_image(
                matching_image,
                OUTPUT_DIR,
                saved
            )

            saved += 1
            last_capture_frame = frame_id

            if saved % 10 == 0 or saved == 1:
                print(
                    f"Saved {saved}/{NUM_IMAGES} | "
                    f"Camera frame: {matching_image.frame} | "
                    f"File: {filename.name}"
                )

        print("\nCollection complete.")
        print(f"Images saved: {saved}/{NUM_IMAGES}")
        print(f"Output folder: {OUTPUT_DIR}")

    except Exception as error:
        print("\nERROR:", repr(error))

    finally:
        print("Cleaning up actors...")

        if camera is not None:
            try:
                camera.stop()
            except Exception:
                pass

            try:
                camera.destroy()
            except Exception:
                pass

        for controller in walker_controllers:
            try:
                controller.stop()
            except Exception:
                pass

        for actor in (
            walker_controllers
            + pedestrians
            + traffic_vehicles
        ):
            try:
                actor.destroy()
            except Exception:
                pass

        if vehicle is not None:
            try:
                vehicle.destroy()
            except Exception:
                pass

        try:
            traffic_manager.set_synchronous_mode(
                original_tm_sync
            )
        except Exception:
            pass

        try:
            world.apply_settings(original_settings)
        except Exception:
            pass

        print("Cleanup complete.")


if __name__ == "__main__":
    main()




