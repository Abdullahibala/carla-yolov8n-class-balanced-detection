import carla

def main():
    # Connect to the running CARLA simulator
    client = carla.Client("localhost", 2000)
    client.set_timeout(20.0)

    # Get the current simulation world
    world = client.get_world()

    # Display simulation information
    print("=" * 50)
    print("CARLA CONNECTION SUCCESSFUL")
    print("=" * 50)

    print("Client version:", client.get_client_version())
    print("Server version:", client.get_server_version())
    print("Current map:", world.get_map().name)

    # Count vehicles and pedestrians
    actors = world.get_actors()

    vehicles = actors.filter("vehicle.*")
    pedestrians = actors.filter("walker.pedestrian.*")

    print("Vehicles currently in the world:", len(vehicles))
    print("Pedestrians currently in the world:", len(pedestrians))

    print("=" * 50)


if __name__ == "__main__":
    main()