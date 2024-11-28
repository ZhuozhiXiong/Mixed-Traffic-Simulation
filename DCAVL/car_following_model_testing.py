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

# import os
# import sys
# try:
#     sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/carla')
# except IndexError:
#     pass
# from agents.navigation.behavior_agent import BehaviorAgent

from automated_agents.intelligent_vehicle import IntelligentVehicle
from automated_agents.intelligent_vehicle import VehicleRegister
from automated_agents.misc import get_speed, get_acceleration


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

def get_speed_profile(leader_speed_list, vehicle_speed_list, delta):
    """
    This function save speed profile to a csv file and plot it with speed limit list
        param vehicle_speed_list: list of vehicle speed in km/h
        param delta: time step in sync mode
    """
    data_path = r'E:\Research\CARLA\experiments\跟驰模型\data'
    time_series = list(range(len(vehicle_speed_list)))
    time_series = np.array(time_series) * delta

    plt.figure()
    plt.plot(time_series, leader_speed_list, linestyle='-', linewidth=1.5, label='Leader speed')
    plt.plot(time_series, vehicle_speed_list, linestyle='-', linewidth=1.5, label='Vehicle speed')
    plt.xlabel('Time(s)')
    plt.ylabel('Speed(km/h)')
    plt.legend()
    plt.show()

    # vehicle_speed_list = np.array(vehicle_speed_list).reshape(-1,1)
    # leader_speed_list = np.array(leader_speed_list).reshape(-1,1)
    # speed_profile = np.concatenate((leader_speed_list,vehicle_speed_list),axis=1)
    # speed_profile_df = pd.DataFrame(speed_profile)
    # speed_profile_df.columns = ['leader','follower']
    # speed_profile_df.to_csv(data_path + '\\vehicle2154_acc.csv', index=False)

def main():
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)

        # Retrieve the world that is currently running
        world = client.get_world()
        map_sim = world.get_map()

        origin_settings = world.get_settings()

        # set sync mode
        delta = 0.1
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = delta
        world.apply_settings(settings)

        blueprint_library = world.get_blueprint_library()
        blueprints = blueprint_library.filter('vehicle.mercedes.coupe')

        # get spawn point
        spawn_point = construct_spawn_points(-1000, 0.5, 0.0)

        # create the blueprint library
        ego_vehicle_bp = blueprints[0]
        if ego_vehicle_bp.has_attribute('color'):
            ego_vehicle_bp.set_attribute('color', '255, 0, 0')
        
        # spawn the vehicle
        vehicle = world.spawn_actor(ego_vehicle_bp, spawn_point)
        print(vehicle.bounding_box.extent.x * 2)

        # spawn a block vehicle
        leader = world.spawn_actor(ego_vehicle_bp, construct_spawn_points(-970, 0.5, 0.0))

        # we need to tick the world once to let the client update the spawn position
        world.tick()

        vehicle_register = VehicleRegister(False,False)

        # create the behavior agent
        int_vehicle = IntelligentVehicle(vehicle, vehicle_register, vehicle_type = 'cav', off_ramp = False, dt = delta)
        agent = int_vehicle.agent

        int_leader = IntelligentVehicle(leader, vehicle_register, vehicle_type = 'hdv', off_ramp = False, dt = delta)
        leader_agent = int_leader.agent

        # set the destination spot
        # destination = construct_spawn_points(320.42, 26.46, 11.97) # off-ramp
        destination = construct_spawn_points(1000, 2, 0) # main stream

        # generate the route
        start_location = vehicle.get_location()
        agent.set_destination(destination.location, start_location, True)
        leader_agent.set_destination(destination.location, leader.get_location(), True)

        trajectory_ngsim = pd.read_csv(r'E:\Research\CARLA\experiments\跟驰模型\data\trajectory_of_vehicle2154.csv')
        vel_ngsim = np.array(trajectory_ngsim['vel']) + 15
        acc_ngsim = np.array(trajectory_ngsim['acc'])
        
        lane_info = []
        vehicle_speed_list = []
        leader_speed_list = []
        speed_limit_list = []
        
        time_step = 0
        i = 0
        follow_flag = False
        while True:
  
            # top view
            spectator = world.get_spectator()
            transform = vehicle.get_transform()
            spectator.set_transform(carla.Transform(transform.location + carla.Location(z=60),
                                                    carla.Rotation(pitch=-90)))
            
            control = leader_agent.run_step()

            if get_speed(leader) / 3.6 >= vel_ngsim[0] or follow_flag:
                if not follow_flag:
                    start_time = time_step
                    follow_flag = True
                i = time_step - start_time
                target_acc = acc_ngsim[i]
                target_vel = vel_ngsim[i]
                planner = leader_agent._local_planner
                # control = planner._vehicle_controller.run_step(planner._target_speed, planner.target_waypoint, None, target_acc)
                control = planner._vehicle_controller.run_step(target_vel*3.6, planner.target_waypoint, None, 1.5)
            # leader.apply_control(control)

            # if time_step % 50 == 0:
            #     print('leader vel:', get_speed(leader) / 3.6)
            #     print('leader acc:',leader_agent._local_planner._vehicle_controller.acc)
            #     print('leader throttle:',control.throttle)
            #     print('leader brake:',control.brake)
            #     print('follow_flag:', follow_flag)

            control = agent.run_step()
            # vehicle.apply_control(control)

            if time_step % 50 == 0:
                print('vel:', get_speed(vehicle) / 3.6)
                print('acc:', get_acceleration(vehicle))
                print('throttle:',control.throttle)
                print('brake:',control.brake)
            
            # get velocity of the ego vehicle in m/s
            leader_speed = get_speed(leader)
            leader_speed_list.append(leader_speed)
            vehicle_speed = get_speed(vehicle)
            vehicle_speed_list.append(vehicle_speed)

            # arrive at the destination
            # if time_step > 2000:
            #     break

            if i >= len(vel_ngsim)-5:
                break

            if len(agent._local_planner._waypoints_queue)<1 or len(leader_agent._local_planner._waypoints_queue)<1:
                print('arrive at the destination successfully, now exit')
                break

            world.tick()
            time_step += 1
            

    finally:
        world.apply_settings(origin_settings)
        vehicle.destroy()
        leader.destroy()

        # get information of velocity
        get_speed_profile(leader_speed_list, vehicle_speed_list, delta)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(' - Exited by user.')