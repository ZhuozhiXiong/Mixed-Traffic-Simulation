# -*- coding: utf-8 -*-

"""Revised automatic control
"""
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: MIT


import random
import carla
import numpy as np
import pandas as pd

# import sys
# import os
# try:
#     sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/carla')
# except IndexError:
#     pass
# from agents.navigation.behavior_agent import BehaviorAgent
# from agents.tools.misc import get_speed

from generate_mixed_traffic_demand import IntelligentVehicle
from automated_agents.misc import get_speed, get_acceleration


def main():
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)

        # Retrieve the world that is currently running
        world = client.get_world()

        origin_settings = world.get_settings()

        tm = client.get_trafficmanager()
        tm_port = tm.get_port()

        # set sync mode
        delta = 0.05
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = delta
        world.apply_settings(settings)

        # read all valid spawn points
        all_default_spawn = world.get_map().get_spawn_points()
        # randomly choose one as the start point
        spawn_point = random.choice(all_default_spawn) if all_default_spawn else carla.Transform()

        # create the blueprint library
        blueprint_library = world.get_blueprint_library()
        # blueprints = blueprint_library.filter('vehicle.*.*')
        # blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
        blueprints = blueprint_library.filter('vehicle.mercedes.coupe')
        # ego_vehicle_bp = random.choice(blueprints)
        ego_vehicle_bp = blueprints[0]

        # spawn the vehicle
        vehicle = world.spawn_actor(ego_vehicle_bp, spawn_point)

        # we need to tick the world once to let the client update the spawn position
        world.tick()

        # # create the behavior agent
        # int_veh = IntelligentVehicle(vehicle, 'cav', False, delta)
        # agent = int_veh.agent
        # # agent = BehaviorAgent(vehicle, 'normal')

        # # set the destination spot
        # spawn_points = world.get_map().get_spawn_points()
        # random.shuffle(spawn_points)

        # # to avoid the destination and start position same
        # if spawn_points[0].location != agent._vehicle.get_location():
        #     destination = spawn_points[0]
        # else:
        #     destination = spawn_points[1]

        # # generate the route
        # start_location = vehicle.get_location()
        # agent.set_destination(destination.location, start_location)

        vehicle.set_autopilot(True,tm_port)

        time = 0
        vehicle_motion = []
        while True:

            time += 1
            if time % 200 == 0:
                print('----------------------------------------')
                vehicle_speed_limit = vehicle.get_speed_limit()
                print(f'speed limit is {vehicle_speed_limit} km/h')
                current_speed = get_speed(vehicle)
                print(f'current speed is {current_speed} km/h')

            # if len(agent._local_planner._waypoints_queue)<5:
            #     random.shuffle(spawn_points)
            #     # to avoid the destination and start position same
            #     if spawn_points[0].location != agent._vehicle.get_location():
            #         destination = spawn_points[0]
            #     else:
            #         destination = spawn_points[1]
            #     # generate the route
            #     agent.set_destination(destination.location)
            #     print('arrive at the destination.')
                
            # top view
            spectator = world.get_spectator()
            transform = vehicle.get_transform()
            spectator.set_transform(carla.Transform(transform.location + carla.Location(z=30),
                                                    carla.Rotation(pitch=-90)))

            # control = agent.run_step()
            # vehicle.apply_control(control)

            velocity = get_speed(vehicle) / 3.6
            acceleration = get_acceleration(vehicle)
            control = vehicle.get_control()
            throttle = control.throttle
            brake = control.brake
            vehicle_motion.append([velocity,acceleration,throttle,brake])

            world.tick()

            if time >= 50000:
                break

    finally:
        world.apply_settings(origin_settings)
        vehicle.destroy()
        data_path = r'E:\Research\CARLA\data'
        if vehicle_motion:
            vehicle_motion = np.array(vehicle_motion)
            vehicle_motion_pd = pd.DataFrame(vehicle_motion)
            # vehicle_motion_pd.to_csv(data_path + '\\vehicle_model_calibration_auto_town04_opt.csv', index=False)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(' - Exited by user.')