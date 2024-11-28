""" This module contains a local planner to perform low-level waypoint following based on PID controllers. """

import numpy as np
from enum import IntEnum
from collections import deque
import random
import carla
from automated_agents.controller import VehicleController
from automated_agents.misc import draw_waypoints, get_speed, is_lane_change


class RoadOption(IntEnum):
    """
    RoadOption represents the possible topological configurations when moving from a segment of lane to other.

    """
    VOID = -1
    LEFT = 1
    RIGHT = 2
    STRAIGHT = 3
    LANEFOLLOW = 4
    CHANGELANELEFT = 5
    CHANGELANERIGHT = 6


class LocalPlanner(object):
    """
    LocalPlanner implements the basic behavior of following a
    trajectory of waypoints that is generated on-the-fly.

    The low-level motion of the vehicle is computed by using two PID controllers,
    one is used for the lateral control and the other for the longitudinal control (cruise speed).

    When multiple paths are available (intersections) this local planner makes a random choice,
    unless a given global plan has already been specified.
    """

    def __init__(self, int_vehicle, map_inst=None):
        """
        :param int_vehicle: actor to apply to local planner logic onto
        :param map_inst: carla.Map instance to avoid the expensive call of getting it.
        """
        self._int_vehicle = int_vehicle
        self._dt = int_vehicle.dt
        self._vehicle = int_vehicle.vehicle
        self._behavior = int_vehicle.behavior
        self._world = self._vehicle.get_world()
        if map_inst:
            if isinstance(map_inst, carla.Map):
                self._map = map_inst
            else:
                print("Warning: Ignoring the given map as it is not a 'carla.Map'")
                self._map = self._world.get_map()
        else:
            self._map = self._world.get_map()

        self._vehicle_controller = None
        self.target_waypoint = None
        self.target_road_option = None

        self._waypoints_queue = deque(maxlen=10000)
        self._min_waypoint_queue_length = 100 # if the length of the waypoint is less than this value, next waypoints are automatically created.
        self._stop_waypoint_creation = True
        self._last_waypoint_removed = None

        # Base parameters
        self._dt = int_vehicle.dt
        self._target_speed = 20.0  # Km/h
        self._sampling_radius = 2.0
        self._offset = 0
        self._base_min_distance = 3.0
        self._distance_ratio = 1 # 0.5
        self._follow_speed_limits = False

        # initializing controller
        self._init_controller()

    def reset_vehicle(self):
        """Reset the ego-vehicle"""
        self._vehicle = None

    def _init_controller(self):
        """Controller initialization"""
        self._vehicle_controller = VehicleController(self._int_vehicle, offset=self._offset)

        # Compute the current vehicle waypoint
        current_waypoint = self._map.get_waypoint(self._vehicle.get_location())
        self.target_waypoint, self.target_road_option = (current_waypoint, RoadOption.LANEFOLLOW)
        self._waypoints_queue.append((self.target_waypoint, self.target_road_option))

    def set_speed(self, speed):
        """
        Changes the target speed

        :param speed: new target speed in Km/h
        :return:
        """
        if self._follow_speed_limits:
            print("WARNING: The max speed is currently set to follow the speed limits. "
                  "Use 'follow_speed_limits' to deactivate this")
        self._target_speed = speed

    def follow_speed_limits(self, value=True):
        """
        Activates a flag that makes the max speed dynamically vary according to the spped limits

        :param value: bool
        :return:
        """
        self._follow_speed_limits = value

    def _compute_next_waypoints(self, k=1):
        """
        Add new waypoints to the trajectory queue.

        :param k: how many waypoints to compute
        :return:
        """
        # check we do not overflow the queue
        available_entries = self._waypoints_queue.maxlen - len(self._waypoints_queue)
        k = min(available_entries, k)

        for _ in range(k):
            last_waypoint = self._waypoints_queue[-1][0]
            next_waypoints = list(last_waypoint.next(self._sampling_radius))

            if len(next_waypoints) == 0:
                break
            elif len(next_waypoints) == 1:
                # only one option available ==> lanefollowing
                next_waypoint = next_waypoints[0]
                road_option = RoadOption.LANEFOLLOW
            else:
                # random choice between the possible options
                road_options_list = _retrieve_options(
                    next_waypoints, last_waypoint)
                road_option = random.choice(road_options_list)
                next_waypoint = next_waypoints[road_options_list.index(
                    road_option)]

            self._waypoints_queue.append((next_waypoint, road_option))

    def set_global_plan(self, current_plan, stop_waypoint_creation=True, clean_queue=True):
        """
        Adds a new plan to the local planner. A plan must be a list of [carla.Waypoint, RoadOption] pairs
        The 'clean_queue` parameter erases the previous plan if True, otherwise, it adds it to the old one
        The 'stop_waypoint_creation' flag stops the automatic creation of random waypoints

        :param current_plan: list of (carla.Waypoint, RoadOption)
        :param stop_waypoint_creation: bool
        :param clean_queue: bool
        :return:
        """
        if clean_queue:
            self._waypoints_queue.clear()

        # Remake the waypoints queue if the new plan has a higher length than the queue
        new_plan_length = len(current_plan) + len(self._waypoints_queue)
        if new_plan_length > self._waypoints_queue.maxlen:
            new_waypoint_queue = deque(maxlen=new_plan_length)
            for wp in self._waypoints_queue:
                new_waypoint_queue.append(wp)
            self._waypoints_queue = new_waypoint_queue

        for elem in current_plan:
            self._waypoints_queue.append(elem)

        self._stop_waypoint_creation = stop_waypoint_creation

    def update_waypoint_queue(self):
        """
        delete waypoints passed by the ego vehicle and calculate possible lane change manuvers
            :return lane_change_list: list of lane change manuvers like 'left' and 'right'
        """
        # Add more waypoints too few in the horizon
        if not self._stop_waypoint_creation and len(self._waypoints_queue) < self._min_waypoint_queue_length:
            self._compute_next_waypoints(k=self._min_waypoint_queue_length)

        # Purge the queue of obsolete waypoints
        veh_location = self._vehicle.get_location()
        vehicle_speed = get_speed(self._vehicle) / 3.6
        self._min_distance = self._base_min_distance + self._distance_ratio * vehicle_speed

        num_waypoint_removed = 0
        for waypoint, _ in self._waypoints_queue:

            if len(self._waypoints_queue) - num_waypoint_removed == 1:
                min_distance = 1  # Don't remove the last waypoint until very close by
            else:
                min_distance = self._min_distance

            if veh_location.distance(waypoint.transform.location) < min_distance:
                num_waypoint_removed += 1
            else:
                break

        lane_change_list = []
        if num_waypoint_removed > 0:
            for _ in range(num_waypoint_removed):
                wp_removed = self._waypoints_queue[0][0]
                lane_change = is_lane_change(self._last_waypoint_removed, wp_removed)
                if lane_change:
                    lane_change_list.append(lane_change)
                self._waypoints_queue.popleft()
                self._last_waypoint_removed = wp_removed
        return lane_change_list

    def run_step(self, target_vehicle, time_gap, debug=False):
        """
        Execute one step of local planning which involves running the longitudinal and lateral PID controllers to
        follow the waypoints trajectory.

        :param debug: boolean flag to activate waypoints debugging
        :return: control, transform, velocity to be applied
        """
        if self._follow_speed_limits:
            self._target_speed = self._vehicle.get_speed_limit()

        # Get the target waypoint and move using controllers. Stop if no target waypoint
        if len(self._waypoints_queue) == 0:
            control = carla.VehicleControl()
            control.steer = 0.0
            control.throttle = 0.0
            control.brake = 1
            control.hand_brake = False
            control.manual_gear_shift = False
            print('there is no waypoint, brake')

            transform = self._vehicle.get_transform()
            velocity = self._vehicle.get_velocity()
            vel_vector = np.array([velocity.x,velocity.y,velocity.z])
            vector_norm = np.linalg.norm(vel_vector)
            if vector_norm > 1e-6:
                velocity.x = velocity.x / vector_norm * max(vector_norm - 5*self._dt, 0)
                velocity.y = velocity.y / vector_norm * max(vector_norm - 5*self._dt, 0)
                velocity.z = velocity.z / vector_norm * max(vector_norm - 5*self._dt, 0)
            else:
                velocity.x = 0.0
                velocity.y = 0.0
                velocity.z = 0.0
        else:
            self.target_waypoint, self.target_road_option = self._waypoints_queue[0]
            control, transform, velocity = self._vehicle_controller.run_step(self._target_speed, self.target_waypoint, target_vehicle, time_gap)

        if debug:
            draw_waypoints(self._vehicle.get_world(), [self.target_waypoint], 1.0)

        return (control, transform, velocity)

    def get_incoming_waypoint_and_direction(self, steps=3):
        """
        Returns direction and waypoint at a distance ahead defined by the user.

            :param steps: number of steps to get the incoming waypoint.
        """
        if len(self._waypoints_queue) > steps:
            return self._waypoints_queue[steps]

        else:
            try:
                wpt, direction = self._waypoints_queue[-1]
                return wpt, direction
            except IndexError as i:
                return None, RoadOption.VOID

    def get_plan(self):
        """Returns the current plan of the local planner"""
        return self._waypoints_queue

    def done(self):
        """
        Returns whether or not the planner has finished

        :return: boolean
        """
        return len(self._waypoints_queue) == 0


def _retrieve_options(list_waypoints, current_waypoint):
    """
    Compute the type of connection between the current active waypoint and the multiple waypoints present in
    list_waypoints. The result is encoded as a list of RoadOption enums.

    :param list_waypoints: list with the possible target waypoints in case of multiple options
    :param current_waypoint: current active waypoint
    :return: list of RoadOption enums representing the type of connection from the active waypoint to each
             candidate in list_waypoints
    """
    options = []
    for next_waypoint in list_waypoints:
        # this is needed because something we are linking to
        # the beggining of an intersection, therefore the
        # variation in angle is small
        next_next_waypoint = next_waypoint.next(3.0)[0]
        link = _compute_connection(current_waypoint, next_next_waypoint)
        options.append(link)

    return options


def _compute_connection(current_waypoint, next_waypoint, threshold=35):
    """
    Compute the type of topological connection between an active waypoint (current_waypoint) and a target waypoint
    (next_waypoint).

    :param current_waypoint: active waypoint
    :param next_waypoint: target waypoint
    :return: the type of topological connection encoded as a RoadOption enum:
             RoadOption.STRAIGHT
             RoadOption.LEFT
             RoadOption.RIGHT
    """
    n = next_waypoint.transform.rotation.yaw
    n = n % 360.0

    c = current_waypoint.transform.rotation.yaw
    c = c % 360.0

    diff_angle = (n - c) % 180.0
    if diff_angle < threshold or diff_angle > (180 - threshold):
        return RoadOption.STRAIGHT
    elif diff_angle > 90.0:
        return RoadOption.LEFT # ---------?
    else:
        return RoadOption.RIGHT
