""" 
This class contains different parameters of behaviors sets for each vehicle type.
params:
    vehicle_type: vehicle type 'hdv' or 'cav'
    max_speed: desired max speed in km/h
    speed_lim_dist: target speed difference from the speed limit in km/h
    min_proximity_threshold: max freespace to detect an obstacle (m)
    braking_distance: if the distance is less than the value, the vehicle will brake (m)
    tailgate_counter: only the value equals to 0, a tailgating can be conducted. set to -1 to deactivate it
    lane_change_counter: only the value equals to 0, a DLC can be conducted. set to -1 to deactivate it
    time_gap: desired time gap (headway) to follow a vehicle (s)
    stand_still_dis: desired rear-end distance when velocity equals to 0 (m)
    politeness_factor: politeness factor in IDM when making lane change decision
    politeness_factor_mlc: politeness factor in IDM when making lane change decision in MLC
    lane_change_threshold: lane_change_threshold in IDM when making lane change decision (m/s2)
"""


class CAV(object):
    """Class for CAV agent."""
    vehicle_type = 'cav'
    max_speed = 120
    min_proximity_threshold = 200
    braking_distance = 5
    tailgate_counter = 0
    lane_change_counter = 100
    time_gap_acc = 1.1
    time_gap_cacc = 0.6
    stand_still_dis = 2

    def __init__(self,speed_lim_dist=5):
        self.speed_lim_dist = speed_lim_dist

class HDV(object):
    """Class for HDV agent."""
    vehicle_type = 'hdv'
    max_speed = 120
    min_proximity_threshold = 200
    braking_distance = 5
    tailgate_counter = 0
    lane_change_counter = 100
    time_gap_hdv = 1.5
    stand_still_dis = 2
    politeness_factor = 1
    politeness_factor_mlc = 0
    lane_change_threshold = 0.1

    def __init__(self,speed_lim_dist=10):
        self.speed_lim_dist = speed_lim_dist
