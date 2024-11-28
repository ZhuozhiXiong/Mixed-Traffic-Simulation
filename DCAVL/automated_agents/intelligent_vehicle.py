import numpy as np
import pandas as pd
import carla
import random
from automated_agents.agent_of_av import AutomatedAgent
from automated_agents.behavior_types import CAV, HDV
from automated_agents.misc import get_speed, get_acceleration, is_within_platoon
from automated_agents.local_planner import RoadOption


class IntelligentVehicle(object):
    """
    This class constructs a intelligent vehicle consisting of a Carla.Vehicle object and 
    an agent controlling the vehicle. Some relevant informations, such as vehicle type, destination
    type, delta in simulation, vehicle register etc., are also recorded.
    """

    def __init__(self, vehicle, vehicle_register, vehicle_type='cav', off_ramp=False, dt=0.05, available_destination=[]):
        """
        Constructor method.

            :param vehicle: a Carla.Vehicle object
            :param vehicle_register: a VehicleRegister class to record information of vehicles in the simulation
            :param vehicle_type: type of the vehicle (str: hdv / cav)
            :param off_ramp: bool for destination of the vehicle, which is designed for off-ramp scenario
            :param dt: delta time in sync mode in simulation
            :param available_destination: list of destinations in the simulation for possible destination changing
        """

        self.vehicle = vehicle
        self.vehicle_type = vehicle_type
        self.off_ramp = off_ramp
        self.dt = dt
        speed_lim_dist = np.random.normal(10, 3) # to form heterogeneous traffic
        if vehicle_type == 'cav':
            self.behavior = CAV(speed_lim_dist)
        else:
            self.behavior = HDV(speed_lim_dist)
        self.vehicle_register = vehicle_register
        agent = AutomatedAgent(self, available_destination=available_destination)
        self.agent = agent
        self.vehicle_register.add_int_vehicle(self)

class VehicleRegister(object):
    """
    This class records information of intelligent vehicles, such as the relationship between vehicle id and vehicle type,
    relationship between vehicle id and agent, vehicle trajectories, list of intelligent vehicle. It also records list
    of platoon with id if platoon control module is activated.
    """

    def __init__(self, delta=0.1, activate_platoon_control=False, activate_vehicle_control=True):
        """
        Constructor method.

            :param activate_platoon_control: bool for whether activate the platoon control module
        """

        self.dt = delta
        self.id_type = {} # {vehicle id: vehicle type}
        self.id_agent = {} # {vehicle id: agent} designed for cooperative control
        self.trajectory = [] # [trajectory information]
        self.int_vehicle_list = [] # [carla.Vehicle]
        self.activate_platoon_control = activate_platoon_control
        self.activate_vehicle_control = activate_vehicle_control

        if activate_platoon_control:
            self.platoon_id_list = [] # [Platoon]
            self.platoon_list = [] # [plaoon id]
            self.id_platoon = {} # {platoon id: platoon}

    def add_int_vehicle(self,int_vehicle):
        """
        add intelligent vehicle to vehicle register in the simulation when a vehicle is spawned
            :param int_vehicle: intelligent vehicle
        """
        vehicle = int_vehicle.vehicle
        vehicle_id = vehicle.id
        vehicle_type = int_vehicle.vehicle_type
        agent = int_vehicle.agent
        if vehicle_id in self.id_type:
            print(f'vehicle {vehicle_id} has been registered on type, skip.')
        else:
            self.id_type[vehicle_id] = vehicle_type
        if vehicle_id in self.id_agent:
            print(f'vehicle {vehicle_id} has been registered on agent, skip.')
        else:
            self.id_agent[vehicle_id] = agent
        self.int_vehicle_list.append(int_vehicle)
    
    def destroy_int_vehicles(self,list_destroy):
        """
        delete information of intelligent vehicles if they are destroied in vehicle register
            :param list_destroy: list of intelligent vehicles
        """
        for int_veh in list_destroy:
            vehicle = int_veh.vehicle
            if self.activate_platoon_control:
                platoon = is_within_platoon(vehicle, self)
                if platoon:
                    print('platoon diverge because of vehicle destruction')
                    platoon.vehicle_diverge(vehicle)
            self.int_vehicle_list.remove(int_veh)
            int_veh.vehicle.destroy()
    
    def destroy_all_int_vehicles(self,client):
        """
        destroy all intelligent vehicles in the simulation and delete all relevant information
            :param client: carla.Client
        """
        print('destroying %d intelligent vehicles.' % (len(self.int_vehicle_list)))
        if self.int_vehicle_list:
            client.apply_batch([carla.command.DestroyActor(x.vehicle) for x in self.int_vehicle_list])

    def upload_information(self,int_vehicle,current_time,lane_id):
        """
        upload trajectory information of each vehicles to the vehicle register in each simlation step
            :param int_vehicle: intelligent vehicle
            :param current_time: current time in the simulation from its beginning
            :param lane_id: current lane id of the vehicle

            trajectory information recorded, which is corresponding to names in function save_trajectory()
                vehile_id: vehicle id
                vehicle_type: vehicle type
                off_ramp: 1 for off-ramp destination, 0 for mainline destination
                lane_id: lane id of the vehicle
                x: position in x axis (m)
                y: position in y axis (m)
                speed: vehicle speed (km/h)
                acceleration: vehicle acceleration (m/s2)
                lane_change
                target_vehicle_id
                target_vehicle_lane_id
                target_vehicle_location.x
                target_vehicle_location.y
                target_vehicle_speed
                target_vehicle_acceleration
                target_vehicle_lane_change
                time: time in the simulation (s)
        """
        vehicle = int_vehicle.vehicle
        vehicle_id = vehicle.id
        vehicle_type_str = int_vehicle.vehicle_type
        if vehicle_type_str == 'cav':
            vehicle_type = 1
        else:
            vehicle_type = 0
        off_ramp = int_vehicle.off_ramp
        if off_ramp:
            off_ramp = 1
        else:
            off_ramp = 0
        location = vehicle.get_location()
        if lane_id == -1 and location.y > 9:
            lane_id = -4
        speed = get_speed(vehicle)
        acceleration = get_acceleration(vehicle)
        agent = int_vehicle.agent
        lane_change = 1 if agent._lane_change_flag else 0
        vehicle_agent = int_vehicle.agent
        platoon_id = -1
        is_leader = -1
        if self.activate_platoon_control:
            if vehicle_id in self.id_platoon:
                platoon_id = self.id_platoon[vehicle_id].platoon_id
                if vehicle_id == self.id_platoon[vehicle_id].leader.id:
                    is_leader = 1
                else:
                    is_leader = 0
        
        preceding_vehicle = vehicle_agent._preceding_vehicle
        rear_vehicle = vehicle_agent._rear_vehicle
        
        if preceding_vehicle:
            preceding_vehicle_id = preceding_vehicle.id
        else:
            preceding_vehicle_id = -1
        if rear_vehicle:
            rear_vehicle_id = rear_vehicle.id
        else:
            rear_vehicle_id = -1

        target_vehicle = vehicle_agent._target_vehicle
        target_vehicle_id = -1
        target_vehicle_lane_id = 0
        target_vehicle_location = carla.Location(x=0,y=0,z=0)
        target_vehicle_speed = 0
        target_vehicle_acceleration = 0
        target_vehicle_lane_change = -1
        target_vehicle_platoon_id = -1
        target_vehicle_is_leader = -1
        if target_vehicle:
            target_vehicle_id = target_vehicle.id
            target_vehicle_lane_id = self.id_agent[target_vehicle_id]._lane_id if target_vehicle_id in self.id_agent else 0
            target_vehicle_location = target_vehicle.get_location()
            if target_vehicle_lane_id == -1 and target_vehicle_location.y > 9:
                target_vehicle_lane_id = -4
            target_vehicle_speed = get_speed(target_vehicle)
            target_vehicle_acceleration = get_acceleration(target_vehicle)
            if target_vehicle_id in self.id_agent:
                target_vehicle_lane_change = 1 if self.id_agent[target_vehicle_id]._lane_change_flag else 0
            else:
                target_vehicle_lane_change = -1
            if self.activate_platoon_control:
                if target_vehicle_id in self.id_platoon:
                    target_vehicle_platoon_id = self.id_platoon[target_vehicle_id].platoon_id
                    if target_vehicle_id == self.id_platoon[target_vehicle_id].leader.id:
                        target_vehicle_is_leader = 1
                    else:
                        target_vehicle_is_leader = 0
            
        self.trajectory.append([vehicle_id,vehicle_type,off_ramp,lane_id,location.x,location.y,speed,acceleration,lane_change,platoon_id,is_leader,
                                target_vehicle_id,target_vehicle_lane_id,target_vehicle_location.x,target_vehicle_location.y,
                                target_vehicle_speed,target_vehicle_acceleration,target_vehicle_lane_change,
                                target_vehicle_platoon_id,target_vehicle_is_leader,preceding_vehicle_id,rear_vehicle_id,current_time])

    def save_trajectory(self, file_name):
        """
        save trajectories recorded by the vehicle register at the end of the simulation to a file
            :param file_name: file name used to save the data
        """
        trajectory = np.array(self.trajectory)
        trajectory = pd.DataFrame(trajectory)
        trajectory.columns = ['vehile_id', 'vehicle_type','off_ramp', 'lane_id', 'x', 'y', 'speed', 'acceleration', 'lane_change','platoon_id','is_leader',
                              'target_vehicle_id','target_vehicle_lane_id','target_vehicle_location_x','target_vehicle_location_y',
                              'target_vehicle_speed','target_vehicle_acceleration','target_vehicle_lane_change',
                              'target_vehicle_platoon_id','target_vehicle_is_leader','preceding_vehicle_id','rear_vehicle_id','time']
        trajectory.to_csv(file_name, index=False)