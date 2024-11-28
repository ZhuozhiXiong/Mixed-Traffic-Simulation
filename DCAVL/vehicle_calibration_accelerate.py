"""
This script spawn a automated vehicle traveling from a start point to an end point.
Information such as road id, section id and lane id of each waypoint in the driving route
is maintained.
"""

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # use tensorflow-cpu

import random
import carla
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import math

# import os
# import sys
# try:
#     sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/carla')
# except IndexError:
#     pass
# from agents.navigation.behavior_agent import BehaviorAgent

from automated_agents.agent_of_av import AutomatedAgent
from generate_mixed_traffic_demand import IntelligentVehicle
from automated_agents.intelligent_vehicle import VehicleRegister
from automated_agents.misc import get_speed, get_acceleration, get_speed_limit


"""
available conmmands
blueprints = blueprint_library.filter('vehicle.tesla.model3')
blueprints = blueprint_library.filter('vehicle.audi.a2')
ego_vehicle_bp = random.choice(blueprints)
"""

def construct_spawn_points(x, y, yaw):
    """
    This function constructs a spawn point in carla.Transform
    based on x, y and yaw
        param x: position in x axis
        param y: position in y axis
        param yaw: yaw value
        return spawn_point: spawn point in carla.Transform
    """
    spawn_point = carla.Transform()
    spawn_point.location.x = x
    spawn_point.location.y = y
    spawn_point.location.z = 1.0
    spawn_point.rotation.pitch = 0.0
    spawn_point.rotation.yaw = yaw
    spawn_point.rotation.roll = 0.0

    return spawn_point

def main():
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)

        # Retrieve the world that is currently running
        world = client.get_world()

        origin_settings = world.get_settings()

        # set sync mode
        delta = 0.1
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = delta
        settings.no_rendering_mode = True
        world.apply_settings(settings)

        blueprint_library = world.get_blueprint_library()
        blueprints = blueprint_library.filter('vehicle.mercedes.coupe')

        # get spawn point
        spawn_point = construct_spawn_points(-4000, -4, 0.09)

        # set the destination spot
        destination = construct_spawn_points(2000, 2, 0) # main stream

        # create the blueprint library
        ego_vehicle_bp = blueprints[0]
        if ego_vehicle_bp.has_attribute('color'):
            ego_vehicle_bp.set_attribute('color', '255, 0, 0')
        
        model_list = []

        vehicle_register = VehicleRegister()

        for throttle in np.arange(0,1,0.01):
        
            # spawn the vehicle
            vehicle = world.spawn_actor(ego_vehicle_bp, spawn_point)

            # we need to tick the world once to let the client update the spawn position
            world.tick()

            # create the behavior agent
            int_vehicle = IntelligentVehicle(vehicle, vehicle_register, vehicle_type = 'cav', off_ramp = False, dt = delta)
            agent = int_vehicle.agent

            # generate the route
            start_location = vehicle.get_location()
            agent.set_destination(destination.location, start_location, True)
            
            time_step = 0
            speed_list = []
            while True:
    
                # top view
                spectator = world.get_spectator()
                transform = vehicle.get_transform()
                spectator.set_transform(carla.Transform(transform.location + carla.Location(z=60),
                                                        carla.Rotation(pitch=-90)))

                control = agent.run_step()
                control.throttle = throttle
                control.brake = 0
                vehicle.apply_control(control)

                vehicle_speed = get_speed(vehicle)
                desired_acc = agent._local_planner._vehicle_controller.acc
                current_acc = get_acceleration(vehicle)
                if time_step > 10:
                    speed_list.append(vehicle_speed)
                    model_list.append([vehicle_speed/3.6, current_acc, throttle, 0])

                if time_step % 50 == 0:
                    print('vel:', get_speed(vehicle) / 3.6)
                    print('desired acc:', desired_acc)
                    print('current acc:', current_acc)
                    print('throttle:', throttle)

                if len(agent._local_planner._waypoints_queue)<1:
                    print('arrive at the destination successfully, now exit')
                    vehicle_register.destroy_int_vehicles([int_vehicle])
                    break

                if vehicle_speed > 120:
                    print('speed reaches 120km/h, now exit')
                    vehicle_register.destroy_int_vehicles([int_vehicle])
                    break

                if time_step > 120 and abs(vehicle_speed - speed_list[-100]) < 0.01:
                    print('speed stays steady, now exit')
                    vehicle_register.destroy_int_vehicles([int_vehicle])
                    break

                world.tick()
                time_step += 1

    finally:
        world.apply_settings(origin_settings)
        vehicle_list = world.get_actors().filter("*vehicle*")
        client.apply_batch([carla.command.DestroyActor(x) for x in vehicle_list])

        # plt.figure()
        # plt.plot(speed_list)
        # plt.show()

        data_path = r'E:\Research\CARLA\data'
        model_list = np.array(model_list)
        model_list_pd = pd.DataFrame(model_list)
        model_list_pd.columns = ['vel','acc','throttle','brake']
        model_list_pd.to_csv(data_path + '\\model_calibration_dataset.csv', index=False)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(' - Exited by user.')