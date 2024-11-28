"""
This script spawn a automated vehicle traveling from a start point to an end point.
Information such as road id, section id and lane id of each waypoint in the driving route
is maintained.
"""

import random
import carla
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from automated_agents.intelligent_vehicle import IntelligentVehicle
from automated_agents.intelligent_vehicle import VehicleRegister
from automated_agents.misc import get_speed


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

def get_speed_profile(vehicle_register):
    """
    This function save speed profile to a csv file and plot it
        param vehicle_register: VehicleRegister object
    """
    data = np.array(vehicle_register.trajectory)
    data = pd.DataFrame(data)
    data.columns = ['vehile_id', 'vehicle_type','off_ramp', 'lane_id', 'x', 'y', 'speed', 'acceleration','time']

    vehicle_id_list = []
    for i in range(len(data['vehile_id'])):
        if data['vehile_id'][i] not in vehicle_id_list:
            vehicle_id_list.append(data['vehile_id'][i])
    
    for id in vehicle_id_list:
        df = data[data['vehile_id']==id]
        speed = np.array(df['speed'])
        time = np.array(df['time'])
        plt.plot(time, speed, linestyle='-', linewidth=1.5)
    plt.xlabel('Time(s)')
    plt.ylabel('Speed(km/h)')
    plt.show()

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
        world.apply_settings(settings)

        blueprint_library = world.get_blueprint_library()
        blueprints = blueprint_library.filter('vehicle.mercedes.coupe')

        # get spawn point
        num_platoon = 10
        front_position_x = -3000
        init_spacing = 16
        spawn_points = []
        for i in range(num_platoon):
            spawn_point = construct_spawn_points(front_position_x-i*init_spacing, -2.5, 0.0)
            spawn_points.append(spawn_point)
        
        destination = construct_spawn_points(2500, 2.04, 0)
        # destination_ramp = construct_spawn_points(320, 26.46, 11.97)

        # create the blueprint library
        ego_vehicle_bp = blueprints[0]
        if ego_vehicle_bp.has_attribute('color'):
            ego_vehicle_bp.set_attribute('color', '255, 0, 0')

        activate_platoon_control = True
        vehicle_register = VehicleRegister(delta, activate_platoon_control,False)

        # vehicle_list = world.get_actors().filter("*vehicle*")
        # client.apply_batch([carla.command.DestroyActor(x) for x in vehicle_list])
        
        # spawn the vehicle
        for i in range(num_platoon):
            vehicle = world.spawn_actor(ego_vehicle_bp, spawn_points[i])
            int_vehicle = IntelligentVehicle(vehicle, vehicle_register, vehicle_type = 'cav', off_ramp = False, dt = delta)
            int_vehicle.agent.set_destination(destination.location, vehicle.get_location(), True)

        # we need to tick the world once to let the client update the spawn position
        world.tick()

        trajectory_ngsim = pd.read_csv(r'E:\Research\CARLA\experiments\跟驰模型\data\trajectory_of_vehicle1367.csv')
        vel_ngsim = np.array(trajectory_ngsim['vel']) + 15
        
        leader = vehicle_register.int_vehicle_list[0]
        vehicle_spect = vehicle_register.int_vehicle_list[2]
        time_step = 0
        index = 0
        follow_flag = False
        while True:
  
            # top view
            spectator = world.get_spectator()
            transform = vehicle_spect.vehicle.get_transform()
            spectator.set_transform(carla.Transform(transform.location + carla.Location(z=100),
                                                    carla.Rotation(pitch=-90)))
            
            # control leader to follow NGSIM trajectory
            if time_step < 100:
                leader.agent._update_information(time_step*delta)
                target_speed = vel_ngsim[0] * 3.6
                local_planner = leader.agent._local_planner
                local_planner.set_speed(target_speed)
                time_gap = 1.5
                leader_control, leader_trans, leader_vel = local_planner.run_step(None,time_gap)
                
            else:
                leader.agent._update_information(time_step*delta)
                planner = leader.agent._local_planner
                time_gap = leader.agent.get_time_gap(None)
                target_vel = vel_ngsim[index]
                leader_control, leader_trans, leader_vel = planner._vehicle_controller.run_step(target_vel*3.6, planner.target_waypoint, None, time_gap)
                index += 1
            
            vehicle_type = leader.agent._vehicle_type
            if vehicle_type == 'cav':
                leader.vehicle.apply_control(leader_control)
            else:
                leader.vehicle.set_transform(leader_trans)
                leader.vehicle.set_target_velocity(leader_vel)

            # leader_control, leader_trans, leader_vel = leader.agent.run_step(time_step*delta)
            # if get_speed(leader.vehicle) / 3.6 >= vel_ngsim[0] or follow_flag:
            #     if not follow_flag:
            #         start_time = time_step
            #         follow_flag = True
            #     index = time_step - start_time
            #     target_vel = vel_ngsim[index]
            #     planner = leader.agent._local_planner
            #     time_gap = leader.agent.get_time_gap(None)
            #     leader_control, leader_trans, leader_vel = planner._vehicle_controller.run_step(target_vel*3.6, planner.target_waypoint, None, time_gap)
            # leader.vehicle.apply_control(leader_control)
            # leader.vehicle.set_transform(leader_trans)
            # leader.vehicle.set_target_velocity(leader_vel)

            # control followers
            for i in range(1,num_platoon):
                vehicle = vehicle_register.int_vehicle_list[i].vehicle
                agent = vehicle_register.int_vehicle_list[i].agent
                vehicle_type = agent._vehicle_type
                if time_step < 100:
                    agent._update_information(time_step*delta)
                    target_speed = vel_ngsim[0] * 3.6
                    local_planner = agent._local_planner
                    local_planner.set_speed(target_speed)
                    time_gap = 1.5
                    control, trans, vel = local_planner.run_step(None,time_gap)
                else:
                    control, trans, vel = agent.run_step(time_step*delta)
                if vehicle_type == 'cav':
                    vehicle.apply_control(control)
                else:
                    vehicle.set_transform(trans)
                    vehicle.set_target_velocity(vel)
            
            # leader_id = leader.vehicle.id
            # if leader_id in vehicle_register.id_platoon:
            #     platoon = vehicle_register.id_platoon[leader_id]
            #     print('member num:', platoon.member_num)

            # arrive at the destination
            if index >= len(vel_ngsim) - 1:
                break

            if len(leader.agent._local_planner._waypoints_queue)<1:
                print('arrive at the destination successfully, now exit')
                break

            world.tick()
            time_step += 1       

    finally:
        world.apply_settings(origin_settings)
        vehicle_register.destroy_all_int_vehicles(client)
        vehicle_register.save_trajectory('E:\Research\CARLA\data\car_following_vehicle1367_platoon03.csv')

        vehicle_list = world.get_actors().filter("*vehicle*")
        client.apply_batch([carla.command.DestroyActor(x) for x in vehicle_list])

        # get information of velocity
        # get_speed_profile(vehicle_register)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(' - Exited by user.')