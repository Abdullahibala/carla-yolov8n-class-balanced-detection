import carla
import random
import time


def main():
    client = carla.Client("localhost", 2000)
    client.set_timeout(20.0)

    world = client.get_world()
    blueprint_library = world.get_blueprint_library()

    # Find available vehicle blueprints
    vehicle_blueprints = blueprint_library.filter("vehicle.*")

    # Select a random vehicle
    vehicle_bp = random.choice(vehicle_blueprints)

    # Get available spawn locations
    spawn_points = world.get_map().get_spawn_points()

    if not spawn_points:
        print("No vehicle spawn points are available.")
        return

    # Find a free spawn point
    vehicle = None

    random.shuffle(spawn_points)

    for spawn_point in spawn_points:
        vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)

        if vehicle is not None:
            break

    if vehicle is None:
        print("Could not spawn a vehicle. Try again.")
        return

    try:
        print("Vehicle spawned successfully!")
        print("Vehicle:", vehicle.type_id)
        print("Actor ID:", vehicle.id)

        # Move the camera to follow the vehicle
        spectator = world.get_spectator()
        vehicle_transform = vehicle.get_transform()

        spectator_transform = carla.Transform(
            vehicle_transform.location + carla.Location(z=30),
            carla.Rotation(pitch=-90)
        )

        spectator.set_transform(spectator_transform)

        print("The spectator camera is positioned above the vehicle.")
        print("The vehicle will remain in the simulation for 60 seconds.")

        time.sleep(60)

    finally:
        # Clean up the vehicle created by this script
        if vehicle is not None:
            vehicle.destroy()
            print("Vehicle removed from the simulation.")


if __name__ == "__main__":
    main()