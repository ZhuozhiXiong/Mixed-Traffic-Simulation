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
from automated_agents.misc import get_speed, compute_distance


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

def get_graph_information(agent):
    """
    This function gets information of the network graph of carla.Map
    from agent of the ago vehicle
        param agent: object of AutomatedAgent
    """
    # plot graph
    graph = agent._global_planner._graph
    nx.draw(graph, with_labels=graph.nodes)
    plt.show()
    
    def get_key(my_dict,val):
        for key, value in my_dict.items():
            if val == value:
                return key
    
        return "There is no such Key"
    
    # get location of a specific node
    node = 30
    location = get_key(agent._global_planner._id_map,node)
    print(f'The location of node {node} is: {location}')

def get_agent_information(agent):
    """
    This function gets information of the agent after it sets
    the global route planner
        param agent: object of AutomatedAgent
    """
    # get global plan
    print('The length of global planner:', len(agent._local_planner._waypoints_queue))
    destination = agent._local_planner._waypoints_queue[-1][0]
    print('end point:', (destination.location.x, destination.location.y))

    # get route junction information
    find_junction = agent._local_planner.target_waypoint.is_junction
    print('The initial state is at junction:', find_junction)

def get_speed_profile(leader_speed_list, vehicle_speed_list, delta):
    """
    This function save speed profile to a csv file and plot it with speed limit list
        param vehicle_speed_list: list of vehicle speed in km/h
        param speed_limit_list: list of speed limit in km/h
        param delta: time step in sync mode
    """
    data_path = r'E:\Research\CARLA\data'
    time_series = list(range(len(vehicle_speed_list)))
    time_series = np.array(time_series) * delta
    vehicle_speed_list = np.array(vehicle_speed_list)
    # vehicle_speed_list_pd = pd.DataFrame(vehicle_speed_list)
    # vehicle_speed_list_pd.to_csv(data_path + '\\speed_profile_mercedes_coupe_bt_throttle0.7.csv', index=False)

    plt.figure()
    plt.plot(time_series, leader_speed_list, linestyle='-', linewidth=1.5, label='Leader speed')
    plt.plot(time_series, vehicle_speed_list, linestyle='-', linewidth=1.5, label='Vehicle speed')
    plt.xlabel('Time(s)')
    plt.ylabel('Speed(km/h)')
    plt.legend()
    plt.show()

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
        spawn_point = construct_spawn_points(-1010, 5.75, 0) # -476

        # create the blueprint library
        ego_vehicle_bp = blueprints[0]
        if ego_vehicle_bp.has_attribute('color'):
            ego_vehicle_bp.set_attribute('color', '255, 178, 102')
        
        # spawn the vehicle
        vehicle = world.spawn_actor(ego_vehicle_bp, spawn_point)

        # spawn a block vehicle
        leader = world.spawn_actor(ego_vehicle_bp, construct_spawn_points(-1000, 2, 0))

        # we need to tick the world once to let the client update the spawn position
        world.tick()

        vehicle_register = VehicleRegister(delta,False,False)

        # create the behavior agent
        int_vehicle = IntelligentVehicle(vehicle, vehicle_register, vehicle_type = 'cav', off_ramp = False, dt = delta)
        agent = int_vehicle.agent

        int_leader = IntelligentVehicle(leader, vehicle_register, vehicle_type = 'cav', off_ramp = True, dt = delta)
        leader_agent = int_leader.agent

        # Get grah information
        get_graph_information(agent)

        # set the destination spot
        destination_ramp = construct_spawn_points(598.05,19.55,6.26) # off-ramp
        destination = construct_spawn_points(1000, 5.54, 0) # main stream

        # generate the route
        start_location = vehicle.get_location()
        agent.set_destination(destination.location, start_location, True)
        leader_agent.set_destination(destination_ramp.location, leader.get_location(), True)

        # get specific information about the agent in the begining
        # get_agent_information(agent)
        
        lane_info = []
        vehicle_speed_list = []
        leader_speed_list = []
        speed_limit_list = []
        
        time_step = 0
        flag_stop = False
        while True:

            # top view
            if leader:
                spectator = world.get_spectator()
                transform = leader.get_transform()
                spectator.set_transform(carla.Transform(transform.location + carla.Location(z=60),
                                                        carla.Rotation(pitch=-90)))

            if leader:
                # if time_step < 2000:
                #     leader.set_target_velocity(carla.Vector3D(x=5,y=0,z=0))
                #     leader.set_target_angular_velocity(carla.Vector3D(x=0,y=0,z=1))
                control, vehicle_transform, velocity = leader_agent.run_step()
                leader.set_transform(vehicle_transform)
                leader.set_target_velocity(velocity)
                # leader.apply_control(control)

            # if time_step > 500:
            #     if not flag_stop:
            #         control.throttle = 0
            #         control.brake = 0
            #         control.hand_brake = False
            #         if get_speed(leader) < 10:
            #             print('deaccelerate enough, now accelerate')
            #             flag_stop = True
            

            # if time_step % 50 == 0:
            #     print('leader vel:', get_speed(leader) / 3.6)
            #     print('leader acc:',leader_agent._local_planner._vehicle_controller.acc)
            #     print('leader throttle:',control.throttle)
            #     print('leader brake:',control.brake)

            if vehicle:
                control, vehicle_transform, velocity = agent.run_step()
                vehicle.set_transform(vehicle_transform)
                vehicle.set_target_velocity(velocity)

            # if time_step % 50 == 0:
            #     print('vel:', get_speed(vehicle) / 3.6)
            #     print('acc:',agent._local_planner._vehicle_controller.acc)
            #     print('throttle:',control.throttle)
            #     print('brake:',control.brake)
                
            # if time_step % 50 == 0:
            #     ego_transform = vehicle.get_transform()
            #     wp = world.get_map().get_waypoint(ego_transform.location, lane_type=carla.LaneType.Any)
            #     print('follower lane id:',wp.lane_id)
            #     ego_front_transform = ego_transform
            #     ego_front_transform.location += carla.Location(
            #         vehicle.bounding_box.extent.x * ego_transform.get_forward_vector())
                
            #     target_transform = leader.get_transform()
            #     wp = world.get_map().get_waypoint(target_transform.location, lane_type=carla.LaneType.Any)
            #     print('leader lane id:',wp.lane_id)
            #     target_rear_transform = target_transform
            #     target_rear_transform.location -= carla.Location(
            #         leader.bounding_box.extent.x * target_transform.get_forward_vector())
            
            #     distance =  compute_distance(target_rear_transform.location, ego_front_transform.location)
            #     print('distance:', distance)
            #     print('follower equ dis:', get_speed(vehicle)/3.6*1.5+2)
            #     print('vel difference:',get_speed(vehicle)/3.6-get_speed(leader)/3.6)
            #     print('--------------------------------------------')

            # get lane information while traveling
            # waypoint = map_sim.get_waypoint(transform.location)
            # info = (waypoint.road_id, waypoint.section_id, waypoint.lane_id)
            # if info in lane_info:
            #     pass
            # else:
            #     lane_info.append(info)
            
            # if time_step % 200 == 0:
            #     print('lane_info:', info)
            #     print('speed:', get_speed(vehicle))
            
            # exam if ego vehicle traveling through a junction
            # find_junction = False
            # if waypoint.is_junction:
            #     find_junction = True
            
            # get velocity of the ago vehicle in m/s
            leader_speed = get_speed(leader)
            leader_speed_list.append(leader_speed)
            vehicle_speed = get_speed(vehicle)
            vehicle_speed_list.append(vehicle_speed)

            # get speed limit
            # speed_limit_list.append(speed_limit)

            # arrive at the destination
            # if time_step > 2000:
            #     break

            if len(agent._local_planner._waypoints_queue)<1 or len(leader_agent._local_planner._waypoints_queue)<1:
                print('arrive at the destination successfully, now exit')
                break

            world.tick()
            time_step += 1
        
        # print information
        # print('The information of DCAVL is as follows:')
        # print(lane_info)
        # print('Does the vehicle traveling through a junction:', find_junction)
    
    finally: 
        world.apply_settings(origin_settings)
        # if vehicle_register:
        #     vehicle_register.destroy_all_int_vehicles(client)
        #     vehicle_register.save_trajectory(r'E:\Research\CARLA\data\vehicle_register_test.csv')
        
        vehicle_list = world.get_actors().filter("*vehicle*")
        client.apply_batch([carla.command.DestroyActor(x) for x in vehicle_list])

        # get information of velocity
        get_speed_profile(leader_speed_list, vehicle_speed_list, delta)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(' - Exited by user.')