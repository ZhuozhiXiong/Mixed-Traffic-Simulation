"""
This script generates mixed traffic with connected automated vehicles and human-driven vehicles in a highway
scenario with dedicated cav lane and off-ramp.
"""

import random
import carla
import numpy as np

# import os
# import sys
# try:
#     sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/carla')
# except IndexError:
#     pass
# from agents.navigation.behavior_agent import BehaviorAgent

from automated_agents.intelligent_vehicle import IntelligentVehicle, VehicleRegister
from automated_agents.misc import get_speed


class MixedTrafficManager(object):
    """
    This class constructs a manager for mixed traffic with human-driven vehicles and connected automated vehicles, which
    can generate vehicles at specific spawn points according to the demand flow volume and MPR, control vehicles based on 
    different agents, and destroy vehicles that reach the destinations or end points.
    """

    def __init__(self, world, delta, start_points, end_points_main, end_points_ramp, lane_type=[],
                 flow_volume=900, flow_volume_rate=0.2, MPR=0.2, activate_platoon_control=False, activate_vehicle_control=False):
        """
        Constructor method.

            :param world: a Carla.World object where the traffic is generated
            :param delta: fixed delta seconds in sync mode (s)
            :param start_points: list of spawn point to generate traffic
            :param end_points_main: list of spawn point to destroy traffic on the mainstream
            :param end_points_ramp: list of spawn point to destroy traffic on the off-ramp
            :param lane_type: define lane type, 1 represents DCAVL, 0 represents normal shared lane
            :param flow_volume: traffic demand (veh/h)
            :param flow_volume_rate: demand rate of going to the ramp (0-1)
            :param MPR: CAV penetration rate (0-1)
            :param activate_platoon_control: activate platoon control module in AutomatedAgent (bool)
        """
        self.vehicle_register = VehicleRegister(delta,activate_platoon_control,activate_vehicle_control)
        self.world = world
        self.delta = delta
        self.start_points = start_points
        self.end_points_main = end_points_main
        self.end_points_ramp = end_points_ramp
        if lane_type:
            self.lane_type = lane_type
        else:
            self.lane_type = [0] * len(start_points)
        self.spawn_points_hdv_index = []
        self.spawn_points_cav_index = []
        for i in range(len(self.start_points)):
            if self.lane_type[i] == 0:
                self.spawn_points_hdv_index.append(i)
            self.spawn_points_cav_index.append(i)
        self.available_destination = end_points_main + end_points_ramp
        self.flow_volume = flow_volume
        self.flow_volume_rate = flow_volume_rate
        self.MPR = MPR
        self.time = 0
        self.volume_calculation = [0] * len(start_points)

        # Get blueprint library
        blueprint_library = self.world.get_blueprint_library()

        # vehicles
        # blueprints = blueprint_library.filter('vehicle.*.*')
        # blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]

        # mercedes
        blueprints = blueprint_library.filter('vehicle.mercedes.coupe')
        
        self.blueprints = blueprints

    def generate_traffic(self):

        # Spawn vehicle with uniform distribution
        total_of_iteration = 3600 / self.delta
        number_of_vehicles = round(total_of_iteration / self.flow_volume)

        # Spawn vehicle if True
        if self.time % number_of_vehicles == 0 and self.time < 5000:

            # Set the vehicle type and its blueprint
            vehile_type_index = random.random()
            if vehile_type_index <= self.MPR:
                vehicle_type = 'cav'
            else:
                vehicle_type = 'hdv'    
            
            # Determine the lane
            if vehicle_type == 'hdv':
                volume = []
                for i in range(len(self.volume_calculation)):
                    if i in self.spawn_points_hdv_index:
                        volume.append(self.volume_calculation[i])
                start_index = self.spawn_points_hdv_index[volume.index(min(volume))]
            else:
                volume = []
                for i in range(len(self.volume_calculation)):
                    if i in self.spawn_points_cav_index:
                        volume.append(self.volume_calculation[i])
                start_index = self.spawn_points_cav_index[volume.index(min(volume))]
            transform = self.start_points[start_index]
            self.volume_calculation[start_index] += 1
            
            # Determine the destination
            ramp_flag = random.random()
            off_ramp = False
            if ramp_flag <= self.flow_volume_rate:
                destination = random.choice(self.end_points_ramp)
                off_ramp = True
            else:
                if vehicle_type == 'hdv':
                    destination = self.end_points_main[start_index]
                else:
                    destination = self.end_points_main[start_index]
            
            blueprint = self.blueprints[0]
            if blueprint.has_attribute('color'):
                if vehicle_type == 'hdv':
                    if off_ramp:
                        color = '0, 200, 0' # hdv go off-ramp: green
                    else:
                        color = '0, 0, 255' # hdv go main road: blue
                else:
                    if off_ramp:
                        color = '255, 178, 102' # cav go off-ramp: orange
                    else:
                        color = '255, 0, 0' # cav go main road: red
                blueprint.set_attribute('color', color)
            vehicle = self.world.spawn_actor(blueprint, transform)

            # Tick the world
            self.world.tick()
            self.time += 1

            # Set intelligent vehicle especially the agent
            intelligent_vehicle = IntelligentVehicle(vehicle,self.vehicle_register,vehicle_type,off_ramp,self.delta,self.available_destination)
            intelligent_vehicle.agent.set_destination(destination.location,vehicle.get_location(),True)
        
        else:
            # Tick the world in sync mode
            self.world.tick()
            self.time += 1

    def control_traffic(self):

        # Control intelligent vehicles
        accelerate_distance = 100
        lane_follow_distance = 400
        if self.vehicle_register.int_vehicle_list:
            for int_veh in self.vehicle_register.int_vehicle_list:
                agent = int_veh.agent
                vehicle = int_veh.vehicle
                vehicle_type = int_veh.vehicle_type

                # warming up to avoid crash
                location_x = vehicle.get_location().x
                if location_x < self.start_points[0].location.x + accelerate_distance:
                    agent._update_information(self.time*self.delta)
                    target_speed = min([agent._behavior.max_speed, agent._speed_limit - 20])
                    local_planner = agent._local_planner
                    local_planner.set_speed(target_speed)
                    time_gap = 1.5
                    _, transform, velocity = local_planner.run_step(None,time_gap)

                    vel_vector = np.array([velocity.x,velocity.y,velocity.z])
                    vector_norm = np.linalg.norm(vel_vector)
                    if vector_norm > 1e-6:
                        velocity.x = velocity.x / vector_norm * target_speed / 3.6
                        velocity.y = velocity.y / vector_norm * target_speed / 3.6
                        velocity.z = velocity.z / vector_norm * target_speed / 3.6
                    
                    vehicle.set_transform(transform)
                    vehicle.set_target_velocity(velocity)
                else:
                    if location_x < self.start_points[0].location.x + accelerate_distance + lane_follow_distance:
                        agent.activate_dlc(False)
                    else:
                        agent.activate_dlc()
                    control, transform, velocity = agent.run_step(self.time*self.delta)
                    if self.vehicle_register.activate_vehicle_control:
                        vehicle.apply_control(control)
                    else:
                        if vehicle_type == 'cav':
                            vehicle.apply_control(control)
                        else:
                            vehicle.set_transform(transform)
                            vehicle.set_target_velocity(velocity)
                

    def destroy_traffic(self):

        # destroy intelligent vehicles
        list_destroy = []
        for int_veh in self.vehicle_register.int_vehicle_list:
            agent = int_veh.agent
            if len(agent._local_planner._waypoints_queue)<5:
                list_destroy.append(int_veh)
        self.vehicle_register.destroy_int_vehicles(list_destroy)

    def run_step(self):
        self.generate_traffic()
        self.control_traffic()
        self.destroy_traffic()

def main():
    try:
        # Connecting the client
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)

        # Retrieve the world that is currently running
        world = client.get_world()

        # Set sync mode
        fixed_delta_seconds = 0.1
        origin_settings = world.get_settings()
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = fixed_delta_seconds
        settings.no_rendering_mode = True
        world.apply_settings(settings)

        # Set the mixed traffic manager
        def construct_spawn_points(x, y, yaw):
            spawn_points = []
            for i in range(len(x)):
                spawn_point = carla.Transform()
                spawn_point.location.x = x[i]
                spawn_point.location.y = y[i]
                spawn_point.location.z = 1.0
                spawn_point.rotation.pitch = 0.0
                spawn_point.rotation.yaw = yaw[i]
                spawn_point.rotation.roll = 0.0
                spawn_points.append(spawn_point)
            return spawn_points

        start_point_x = [-3000, -3000, -3000]
        start_point_y = [-2.5, 1.25, 5]
        start_point_yaw = [0.09, 0.09, 0.09]

        start_points = construct_spawn_points(start_point_x,start_point_y,start_point_yaw)
        
        end_point_main_x = [2100, 2100, 2100]
        end_point_main_y = [2.04, 5.79, 9.54]
        end_point_main_yaw = [0.0, 0.0, 0.0]

        end_points_main = construct_spawn_points(end_point_main_x,end_point_main_y,end_point_main_yaw)
        
        end_point_ramp_x = [598.05]
        end_point_ramp_y = [19.55]
        end_point_ramp_yaw = [6.26]

        end_points_ramp = construct_spawn_points(end_point_ramp_x,end_point_ramp_y,end_point_ramp_yaw)

        flow_volume = 3600
        flow_volume_rate = 0.2
        MPR = 1
        activate_platoon_control = False
        activate_vehicle_control = False
        lane_type = [0, 0, 0]
        # random.seed(0)
        mixed_traffic_manager = MixedTrafficManager(world,fixed_delta_seconds,start_points,end_points_main,
                                                    end_points_ramp,lane_type,flow_volume,flow_volume_rate,MPR,
                                                    activate_platoon_control,activate_vehicle_control)
        
        # spectator = world.get_spectator()
        # transform = start_points[0]
        # spectator.set_transform(carla.Transform(transform.location + carla.Location(x=0,z=60),
        #                                         carla.Rotation(pitch=-90)))

        vehicle_list = world.get_actors().filter("*vehicle*")
        client.apply_batch([carla.command.DestroyActor(x) for x in vehicle_list])

        # Start the simulation
        simluation_step = 0
        while simluation_step < 10000:

            # Manage mixed traffic
            mixed_traffic_manager.run_step()
            simluation_step += 1

            if simluation_step % 100 == 0:
                print('Simulation step:',simluation_step)
    
    finally:

        # Return the world to the original settings
        world.apply_settings(origin_settings)

        # Destroy the remaining vehicles
        if mixed_traffic_manager:
            file_name = r'E:\Research\CARLA\data\mixed_traffic_flow48.csv'
            mixed_traffic_manager.vehicle_register.destroy_all_int_vehicles(client)
            mixed_traffic_manager.vehicle_register.save_trajectory(file_name)
        
        vehicle_list = world.get_actors().filter("*vehicle*")
        client.apply_batch([carla.command.DestroyActor(x) for x in vehicle_list])

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(' - Exited by user.')
