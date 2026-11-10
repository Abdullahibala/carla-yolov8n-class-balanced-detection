
import carla
import os
import time
import random
import queue

# Folder where images will be saved
OUTPUT_DIR = "carla_dataset/images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Number of images to capture
NUM_IMAGES = 100


def main():
    client = carla.Client("localhost", 2000)
    client.set_timeout(20.0)

    world = client.get_world()
    blueprint_library = world.get_blueprint_library()

    vehicle = None
    camera = None
    image_queue = queue.Queue()

    try:
        # Choose a vehicle
        vehicle_bp = blueprint_library.find("vehicle.tesla.model3")

        # Find an available spawn point
        spawn_points = world.get_map().get_spawn_points()
        random.shuffle(spawn_points)

        for spawn_point in spawn_points:
            vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
            if vehicle is not None:
                break

        if vehicle is None:
            print("Could not spawn a vehicle. Try again.")
            return

        print("Vehicle spawned:", vehicle.type_id)

        # Create an RGB camera
        camera_bp = blueprint_library.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", "1280")
        camera_bp.set_attribute("image_size_y", "720")
        camera_bp.set_attribute("fov", "90")
        camera_bp.set_attribute("sensor_tick", "0.1")

        # Attach the camera near the front windshield
        camera_transform = carla.Transform(
            carla.Location(x=1.5, z=1.6)
        )

        camera = world.spawn_actor(
            camera_bp,
            camera_transform,
            attach_to=vehicle
        )

        # Save captured images through a queue
        camera.listen(image_queue.put)

        print("RGB camera attached.")
        print("Capturing images...")

        for i in range(NUM_IMAGES):
            image = image_queue.get(timeout=10.0)

            filename = os.path.join(
                OUTPUT_DIR,
                f"image_{i:05d}.png"
            )

            image.save_to_disk(filename)

            if (i + 1) % 10 == 0:
                print(f"Saved {i + 1}/{NUM_IMAGES} images")

        print(f"\nFinished! Images saved in: {OUTPUT_DIR}")

    except queue.Empty:
        print("Timed out waiting for a camera image.")
    except Exception as e:
        print("Error:", e)

    finally:
        # Stop the camera before destroying actors
        if camera is not None:
            camera.stop()
            camera.destroy()

        if vehicle is not None:
            vehicle.destroy()

        print("Simulation actors cleaned up.")


if __name__ == "__main__":
    main()
