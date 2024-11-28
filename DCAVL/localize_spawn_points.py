"""
This script calculates the position of the spawn points.
"""

import carla

spawn_point_index = 0

def main():

    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    # Retrieve the world that is currently running
    world = client.get_world()

    # read all valid spawn points
    all_default_spawn = world.get_map().get_spawn_points()
    spawn_point = all_default_spawn[spawn_point_index]
    print('location', [spawn_point.location.x, spawn_point.location.y, spawn_point.location.z])
    print('rotation', [spawn_point.rotation.pitch, spawn_point.rotation.yaw, spawn_point.rotation.roll])


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(' - Exited by user.')