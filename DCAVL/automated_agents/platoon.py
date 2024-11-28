import numpy as np
from automated_agents.misc import get_speed, get_acceleration, compute_distance, is_within_distance, is_within_platoon


class Platoon(object):
    """
    This class constructs a vehicular platoon with CAVs on DCAVL, which has several features such as
    leader, members, information flow topology, desired time gap, longitudinal controllers. Some manuvers
    can be conducted in the simulation such as two platoons merge to a platoon, vehicle merges into a
    platoon, a member diverges from a platoon etc.
    """
    
    def __init__(self,leader,follower,vehicle_register):
        """
        Constructor method.

            :param leader: a Carla.Vehicle object to lead a platoon
            :param follower: a Carla.Vehicle object to follow the leader
            :param vehicle_register: vehicle register to save and get information in the simulation
        """
        self.vehicle_register = vehicle_register
        self.dt = vehicle_register.dt
        platoon_id = max(vehicle_register.platoon_id_list)+1 if vehicle_register.platoon_id_list else 0
        self.platoon_id = platoon_id
        self.leader = leader
        self.members = []
        self.members.append(leader)
        self.members.append(follower)
        self.member_num = len(self.members)
        self.topology = self.construct_topology()
        self.time_gap = 0.3 # n1=0.6, n3=0.2
        self.time_gap_cacc = 0.6
        self.stand_still_dis = 2
        self.vehicle_length = 5
        self.max_member_num = 10
        self.target_speed = 120
        self.update_target_speed()
        self.kp = 0.1 # n1=0.1, n3=0.1
        self.kv = 1.65 # n1=1.65, n3=1.67
        self.ka = 0.51 # n1=0.51, n3=0.84
        self.update_info_in_register()
    
    def update_target_speed(self):
        """
        This module regards target speed of the leader as that of the platoon
        """
        leader_id = self.leader.id
        id_agent = self.vehicle_register.id_agent
        if leader_id in id_agent:
            agent = id_agent[leader_id]
            self.target_speed = agent._target_speed
        else:
            self.target_speed = 120

    def platoon_merge(self,front_platoon):
        """
        This module manages information of a platoon if another platoon plans to merge
            :param front_platoon: the preceding platoon
            :return bool: if the maneuver is conducted
        """
        front_location = front_platoon.members[-1].get_location()
        rear_location = self.leader.get_location()
        dis = compute_distance(front_location,rear_location)
        equ_dis = self.stand_still_dis + self.vehicle_length + self.time_gap_cacc * get_speed(self.leader)/3.6
        if dis > 2*equ_dis:
            return False
        
        front_members = front_platoon.members
        rear_members = self.members
        if len(front_members) + len(rear_members) > self.max_member_num:
            return False
        else:
            for rear_vehicle in rear_members:
                front_members.append(rear_vehicle)
            self.destroy_platoon()
            front_platoon.member_num = len(front_platoon.members)
            front_platoon.topology = front_platoon.construct_topology()
            front_platoon.update_info_in_register()
            print('platoon merge')
            return True

    def vehicle_merge(self,vehicle):
        """
        This module manages information of a platoon if a vehicle plans to merge
            :param vehicle: a preceding or rear vehicle
            :return bool: if the maneuver is conducted
        """
        if vehicle.id in self.vehicle_register.id_agent:
            target_off_ramp = self.vehicle_register.id_agent[vehicle.id]._off_ramp
        else:
            target_off_ramp = True
        if target_off_ramp:
            return False
        
        vehicle_transform = vehicle.get_transform()
        leader_transform = self.leader.get_transform()
        rear_transform = self.members[-1].get_transform()
        is_in_the_front = is_within_distance(vehicle_transform, leader_transform, 300, angle_interval=[0, 90])
        is_in_the_rear = is_within_distance(vehicle_transform, rear_transform, 300, angle_interval=[90, 180])

        vehicle_location = vehicle.get_location()
        leader_location = self.leader.get_location()
        rear_location = self.members[-1].get_location()
        front_distance = compute_distance(vehicle_location,leader_location)
        rear_distance = compute_distance(vehicle_location,rear_location)

        if is_in_the_rear:
            equ_dis = self.stand_still_dis + self.vehicle_length + self.time_gap_cacc * get_speed(vehicle)/3.6
            if rear_distance >= 2*equ_dis:
                return False
        elif is_in_the_front:
            equ_dis = self.stand_still_dis + self.vehicle_length + self.time_gap_cacc * get_speed(self.leader)/3.6
            if front_distance >= 2*equ_dis:
                return False
            
        if self.member_num >= self.max_member_num:
            return False
        
        if is_in_the_front:
            self.leader = vehicle
            self.members.insert(0,vehicle)
        elif is_in_the_rear:
            self.members.append(vehicle)
        else:
            for i in range(1,self.member_num):
                transform = self.members[i].get_transform()
                in_the_front = is_within_distance(vehicle_transform, transform, 300, angle_interval=[0, 90])
                if in_the_front:
                    pos = i
                    break
            self.members.insert(pos,vehicle)
        self.member_num += 1
        self.topology = self.construct_topology()
        self.update_info_in_register()
        self.update_target_speed()
        print('vehicle merge')
        return True

    def vehicle_diverge(self,vehicle):
        """
        This module manages information of a platoon if a member plans to diverge
            :param vehicle: a member plans to diverge
        """
        if self.member_num <= 2:
            self.destroy_platoon()
        else:
            if vehicle.id == self.leader.id:
                self.leader = self.members[1]
            for v in self.members:
                if v.id == vehicle.id:
                    self.members.remove(v)
                    self.vehicle_register.id_platoon.pop(vehicle.id)
            self.member_num -= 1
            self.topology = self.construct_topology()
            self.update_info_in_register()
            self.update_target_speed()
            print('vehicle diverge')

    def update_info_in_register(self):
        """
        This module updates information of a platoon on the vehicle register
        """
        if self.platoon_id not in self.vehicle_register.platoon_id_list:
            self.vehicle_register.platoon_id_list.append(self.platoon_id)
        if self not in self.vehicle_register.platoon_list:
            self.vehicle_register.platoon_list.append(self)
        for vehicle in self.members:
            if vehicle.id not in self.vehicle_register.id_platoon:
                self.vehicle_register.id_platoon[vehicle.id] = self

    def construct_topology(self):
        """
        This module constructs the information flow topology matrix of a platoon based on its member number
        """
        topology = np.zeros((self.member_num,self.member_num))
        for i in range(self.member_num):
            for j in range(self.member_num):
                # if j == i-1:
                #     topology[i,j] = 1
                if j in [i-1, i-2, i-3]:
                    topology[i,j] = 1
        return topology
    
    def destroy_platoon(self):
        """
        This module deletes information of a paltoon that is not exist in the vehicle register
        """
        print('platoon destroy')
        self.vehicle_register.platoon_id_list.remove(self.platoon_id)
        self.vehicle_register.platoon_list.remove(self)
        for vehicle in self.members:
            self.vehicle_register.id_platoon.pop(vehicle.id)
    
    def run_step(self,vehicle,control, transform, vel_vector3d):
        """
        This module controls car-following behavior of a vehicle in the platoon
            :param vehicle: vehicle to control
            :return (throttle, brake): value of throttle and brake of carla.VehicleControl
        """
        vehicle_id = vehicle.id
        for v in range(self.member_num):
            if vehicle_id == self.members[v].id:
                i = v
        equ_dis = self.stand_still_dis + self.vehicle_length + self.time_gap*get_speed(vehicle)/3.6
        position = [x.get_location().x for x in self.members]
        velocity = [get_speed(x)/3.6 for x in self.members]
        acceleration = [get_acceleration(x) for x in self.members]
        acc = 0
        for j in range(self.member_num):
            acc += self.topology[i,j]*(self.kp*(position[j]-position[i]-equ_dis*(i-j))+
                                          self.kv*(velocity[j]-velocity[i])+self.ka*(acceleration[j]-acceleration[i]))
        acc = acc / sum(self.topology[i,:])
        def calculate_output(x,y):
            return 0.2613+0.006933*x+0.04656*y+0.00595*x*y+0.002213*y**2
        
        current_vel = get_speed(vehicle) / 3.6
        if current_vel >= 15:
            acc = np.clip(acc,-5,2)
        else:
            acc = np.clip(acc,-5,5)
        if current_vel >= min(self.target_speed + 10, 120) / 3.6:
            acc = 0
        output = calculate_output(current_vel, acc)
        if output >= 0:
            control.throttle = np.clip(output, 0.0, 0.7)
            control.brake = 0.0
        else:
            control.throttle = 0.0
            control.brake = np.clip(-output, 0.0, 0.85)
        
        acc = np.clip(acc,-5,2)
        target_vel = current_vel + acc * self.dt
        vel_vector = np.array([vel_vector3d.x,vel_vector3d.y,vel_vector3d.z])
        vector_norm = np.linalg.norm(vel_vector)
        if vector_norm > 1e-6:
            vel_vector3d.x = vel_vector3d.x / vector_norm * target_vel
            vel_vector3d.y = vel_vector3d.y / vector_norm * target_vel
            vel_vector3d.z = vel_vector3d.z / vector_norm * target_vel
        
        return (control, transform, vel_vector3d)