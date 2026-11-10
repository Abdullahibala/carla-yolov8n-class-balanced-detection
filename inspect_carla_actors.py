import carla
from collections import Counter

# ============================================================
# CONFIGURATION
# ============================================================

HOST = "127.0.0.1"
PORT = 2000

# ============================================================
# CONNECT TO CARLA
# ============================================================

client = carla.Client(HOST, PORT)
client.set_timeout(20.0)

world = client.get_world()
blueprints = world.get_blueprint_library()
actors = world.get_actors()

print("=" * 70)
print("CARLA ACTOR AND BLUEPRINT AUDIT")
print("=" * 70)

print(f"\nCARLA version: {client.get_server_version()}")
print(f"Current map: {world.get_map().name}")

# ============================================================
# 1. INSPECT VEHICLE BLUEPRINTS
# ============================================================

print("\n" + "=" * 70)
print("1. AVAILABLE VEHICLE BLUEPRINTS")
print("=" * 70)

vehicle_blueprints = sorted(
    blueprints.filter("vehicle.*"),
    key=lambda bp: bp.id
)

for bp in vehicle_blueprints:
    if bp.has_attribute("number_of_wheels"):
        wheels = bp.get_attribute(
            "number_of_wheels"
        ).as_int()
    else:
        wheels = "Unknown"

    print(f"{bp.id:55s} Wheels: {wheels}")

print(f"\nTotal vehicle blueprints: {len(vehicle_blueprints)}")

# ============================================================
# 2. COUNT ACTORS CURRENTLY IN THE WORLD
# ============================================================

print("\n" + "=" * 70)
print("2. RELEVANT ACTORS CURRENTLY IN THE WORLD")
print("=" * 70)

actor_counts = Counter(
    actor.type_id
    for actor in actors
    if (
        actor.type_id.startswith("vehicle.")
        or actor.type_id.startswith("walker.")
        or actor.type_id.startswith("traffic.")
    )
)

if actor_counts:
    for actor_type, count in sorted(actor_counts.items()):
        print(f"{actor_type:55s} Count: {count}")
else:
    print("No relevant vehicle, pedestrian, or traffic actors found.")

print(f"\nTotal relevant actor instances: {sum(actor_counts.values())}")

# ============================================================
# 3. INSPECT TRAFFIC-RELATED BLUEPRINTS
# ============================================================

print("\n" + "=" * 70)
print("3. TRAFFIC-RELATED BLUEPRINTS")
print("=" * 70)

traffic_blueprints = sorted(
    [
        bp for bp in blueprints
        if bp.id.startswith("traffic.")
    ],
    key=lambda bp: bp.id
)

if traffic_blueprints:
    for bp in traffic_blueprints:
        print(bp.id)
else:
    print("No blueprints beginning with 'traffic.' were found.")

print(f"\nTotal traffic-related blueprints: {len(traffic_blueprints)}")

# ============================================================
# 4. SUMMARIZE POTENTIAL VEHICLE CATEGORIES
# ============================================================

print("\n" + "=" * 70)
print("4. POTENTIAL VEHICLE CATEGORIES")
print("=" * 70)

categories = {
    "Motorcycles": [],
    "Bicycles": [],
    "Trucks": [],
    "Buses": [],
    "Other vehicles": [],
}

for bp in vehicle_blueprints:
    blueprint_id = bp.id.lower()
    name = blueprint_id.split(".")[-1]

    if any(
        word in name
        for word in [
            "motorcycle",
            "harley",
            "yamaha",
            "kawasaki",
            "vespa",
            "diamondback"
        ]
    ):
        categories["Motorcycles"].append(bp.id)

    elif any(
        word in name
        for word in [
            "bicycle",
            "bh.crossbike",
            "gazelle"
        ]
    ):
        categories["Bicycles"].append(bp.id)

    elif any(
        word in name
        for word in [
            "truck",
            "firetruck",
            "carlacola",
            "cybertruck"
        ]
    ):
        categories["Trucks"].append(bp.id)

    elif any(
        word in name
        for word in [
            "bus",
            "coach"
        ]
    ):
        categories["Buses"].append(bp.id)

    else:
        categories["Other vehicles"].append(bp.id)

for category, items in categories.items():
    print(f"\n{category}: {len(items)}")

    for item in items:
        print(f"  - {item}")

# ============================================================
# 5. SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("AUDIT SUMMARY")
print("=" * 70)

print(f"Vehicle blueprints: {len(vehicle_blueprints)}")
print(f"Traffic-related blueprints: {len(traffic_blueprints)}")
print(f"Actors currently in the world: {len(actors)}")

print("\nAudit complete.")