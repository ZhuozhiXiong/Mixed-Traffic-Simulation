"""
This script spawn a vehicle in a specific point which is presented by index of all the
spawn points in the world or a specific location and rotation.
"""

import random
import carla


def main():
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)

        # Retrieve the world that is currently running
        world = client.get_world()

        origin_settings = world.get_settings()

        # set sync mode
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)

        blueprint_library = world.get_blueprint_library()
        blueprint_library = blueprint_library.filter('vehicle.*.*')
        blueprint_library = [x for x in blueprint_library if int(x.get_attribute('number_of_wheels')) == 4]

        # find a specific spawn point with a index
        # spawn_point_index = 0
        # all_default_spawn = world.get_map().get_spawn_points()
        # spawn_point = all_default_spawn[spawn_point_index]
        
        # find a specific spawn point with location and rotation
        spawn_point = carla.Transform()
        spawn_point.location.x = 320.42
        spawn_point.location.y = 26.46
        spawn_point.location.z = 4.0
        spawn_point.rotation.pitch = 0.0
        spawn_point.rotation.yaw = 11.97
        spawn_point.location.roll = 0.0

        # create the blueprint library
        ego_vehicle_bp = random.choice(blueprint_library)
        ego_vehicle_bp.set_attribute('color', '0, 0, 0')
        # spawn the vehicle
        vehicle = world.spawn_actor(ego_vehicle_bp, spawn_point)

        # we need to tick the world once to let the client update the spawn position
        world.tick()

        while True:
            world.tick()

    finally:
        world.apply_settings(origin_settings)
        vehicle.destroy()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(' - Exited by user.')