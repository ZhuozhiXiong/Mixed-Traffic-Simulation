"""
This module implements an agent that roams around a track following waypoints from
the global planner and avoiding other vehicles. The agent also responds to traffic lights,
traffic signs, and has different possible configurations.
There are several fuctions or modules need to be defined in a specific map:
    
    Modules in AutomatedAgent:
    self._activate_specific_map_speed_limit: define speed limit
    lane_change_in_off_ramp: regard lane_id -4 as dedicated lane before off-ramp
    lane_change_in_off_ramp_judgement: define area to change lane for off-ramp
        test map: off_ramp_influence_range_x = [0, 200]
        large map: off_ramp_influence_range_x = [0, 450]
    go_to_next_destination: define relation between lane_id and destination
    change_to_available_destination: define relation between lane_id and destination
    discretionary_lane_change_judgement: judge_one_side: cannot change to lane -4 if it is not go to off-ramp
    form_platoon_judgement: check range for dedicated CAV lane
        test map: y_bound = 5
        large map: y_bound = 9

    Functions in automated_agents.misc:
    get_speed_limit: define speed limit
    dcavl_permit: define CAV dedicated lane information
"""

import carla
import random
import numpy as np
from shapely.geometry import Polygon

from automated_agents.local_planner import LocalPlanner, RoadOption
from automated_agents.global_route_planner import GlobalRoutePlanner
from automated_agents.platoon import Platoon
from automated_agents.misc import (get_speed, is_within_distance, get_trafficlight_trigger_location, 
                                   compute_distance, dcavl_permit, get_acceleration, get_speed_limit)


class AutomatedAgent(object):
    """
    AutomatedAgent implements an agent that navigates scenes to reach a given
    target destination, by computing the shortest possible path to it.
    This agent can correctly follow traffic signs, speed limitations,
    traffic lights, while also taking into account nearby vehicles. Lane changing
    decisions can be taken by analyzing the surrounding environment. Besides, it can
    perform cooperative behaviors like platoon control with other vehicle if it is a connected
    automated vehicle.
    """

    def __init__(self, int_vehicle, map_inst=None, grp_inst=None, available_destination=[]):
        """
        Initialization the agent paramters, the local and the global planner, and some function modules.

            :param int_vehicle: actor to apply to agent logic onto
            :param map_inst: carla.Map instance to avoid the expensive call of getting it.
            :param grp_inst: GlobalRoutePlanner instance to avoid the expensive call of getting it.
            :param available_destination: list of carla.Transform as vehicle destinations off_ramp_influence_range_x = [0, 200]

        """
        self._int_vehicle = int_vehicle
        self._dt = int_vehicle.dt
        self._vehicle_type = int_vehicle.vehicle_type
        self._vehicle = int_vehicle.vehicle
        self._id = self._vehicle.id
        self._off_ramp = int_vehicle.off_ramp
        self._behavior = int_vehicle.behavior
        self._vehicle_register = int_vehicle.vehicle_register
        self._activate_platoon_control = self._vehicle_register.activate_platoon_control
        self._available_destination = available_destination
        self._off_ramp_influence_range_x = [-500, 490]
        
        self._world = self._vehicle.get_world()
        if map_inst:
            if isinstance(map_inst, carla.Map):
                self._map = map_inst
            else:
                print("Warning: Ignoring the given map as it is not a 'carla.Map'")
                self._map = self._world.get_map()
        else:
            self._map = self._world.get_map()
        self._last_traffic_light = None

        # Base parameters
        self._ignore_traffic_lights = False
        self._ignore_stop_signs = False
        self._ignore_vehicles = False
        self._use_bbs_detection = False
        self._deactivate_bbs_detection = True # don't use bounding box to detect surrounding vehicles even if in intersections
        self._deactivate_dlc = False # don't perform discretionary lane change
        self._deactivate_mandatory_lane_change = True # don't change lane according to the route
        self._activate_specific_map_speed_limit = True # use speed limit designed in a specific map
        self._activate_change_destination = True # if a vehicle has to go off-ramp but there is no chance to change lane, then it can change destination to avoid block the way
        self._activate_deceleration_off_ramp = True # decelerate when tries to fine possible gaps in off-ramp influence area
        self._target_speed = 0
        self._sampling_resolution = 2.0
        self._base_tlight_threshold = 5.0  # meters
        self._base_vehicle_threshold = self._behavior.stand_still_dis  # max freespace to check for obstacles (meters)
        self._max_brake = 0.5 # brake to give a emergency stop
        self._offset = 0
        self._emergency_stop_period_threshold = 3 # if stop longer than the threshold, it has to go ahead to avoid blocking the way

        # Initialize the planners
        self._local_planner = LocalPlanner(int_vehicle, map_inst=self._map)
        if grp_inst:
            if isinstance(grp_inst, GlobalRoutePlanner):
                self._global_planner = grp_inst
            else:
                print("Warning: Ignoring the given planner as it is not a 'GlobalRoutePlanner'")
                self._global_planner = GlobalRoutePlanner(self._map, self._sampling_resolution)
        else:
            self._global_planner = GlobalRoutePlanner(self._map, self._sampling_resolution)

        # Get the static elements of the scene
        self._lights_list = self._world.get_actors().filter("*traffic_light*")
        self._lights_map = {}  # Dictionary mapping a traffic light to a wp corrspoing to its trigger volume location

        self._look_ahead_steps = 3

        # Vehicle information
        self._speed = 0
        self._speed_limit = 0
        self._direction = None
        self._target_vehicle = None
        self._preceding_vehicle = None
        self._rear_vehicle = None
        self._incoming_direction = None
        self._incoming_waypoint = None
        self._lane_change_flag = False # if it is changing lane
        self._lane_change_direction = 0
        self._emergency_stop_period = 0 # record how long it has stopped
        self._remaining_lane_change = [] # list of remaining lane change 'left' or 'right' recorded by local planner, there remains a bug if changing destination in the trip
        self._safe_acc_ratio = 1
        if self._vehicle_type == 'cav':
            self._safe_acc = - self._safe_acc_ratio * 4
        else:
            self._safe_acc = - self._safe_acc_ratio * 4
        self._lambda_factor = 0.4

        location = self._vehicle.get_location()
        self._vehicle_wp = self._map.get_waypoint(location)
        self._lane_id = self._vehicle_wp.lane_id
        self._last_lane_id = self._lane_id
        self._target_lane_id = self._lane_id # target lane id when a lane change manuver is conducted
        self._change_destination = False

        
    def set_target_speed(self, speed):
        """
        Changes the target speed of the agent
            :param speed (float): target speed in Km/h
        """
        self._target_speed = speed
        self._local_planner.set_speed(speed)

    def follow_speed_limits(self, value=True):
        """
        If active, the agent will dynamically change the target speed according to the speed limits

            :param value (bool): whether or not to activate this behavior
        """
        self._local_planner.follow_speed_limits(value)

    def get_local_planner(self):
        """Get method for protected member local planner"""
        return self._local_planner

    def get_global_planner(self):
        """Get method for protected member local planner"""
        return self._global_planner

    def set_destination(self, end_location, start_location=None, clean_queue=True):
        """
        This method creates a list of waypoints between a starting and ending location,
        based on the route returned by the global router, and adds it to the local planner.
        If no starting location is passed, the vehicle local planner's target location is chosen,
        which corresponds (by default), to a location about 5 meters in front of the vehicle.

            :param end_location (carla.Location): final location of the route
            :param start_location (carla.Location): starting location of the route
        """
        if not start_location:
            start_location = self._local_planner.target_waypoint.transform.location

        start_waypoint = self._map.get_waypoint(start_location)
        end_waypoint = self._map.get_waypoint(end_location)

        route_trace = self.trace_route(start_waypoint, end_waypoint)
        self._local_planner.set_global_plan(route_trace, clean_queue=clean_queue)
        self._local_planner.update_waypoint_queue()

        self._remaining_lane_change = []

    def set_global_plan(self, plan, stop_waypoint_creation=True, clean_queue=True):
        """
        Adds a specific plan to the agent.

            :param plan: list of [carla.Waypoint, RoadOption] representing the route to be followed
            :param stop_waypoint_creation: stops the automatic random creation of waypoints
            :param clean_queue: resets the current agent's plan
        """
        self._local_planner.set_global_plan(
            plan,
            stop_waypoint_creation=stop_waypoint_creation,
            clean_queue=clean_queue
        )

    def trace_route(self, start_waypoint, end_waypoint):
        """
        Calculates the shortest route between a starting and ending waypoint.

            :param start_waypoint (carla.Waypoint): initial waypoint
            :param end_waypoint (carla.Waypoint): final waypoint
        """
        start_location = start_waypoint.transform.location
        end_location = end_waypoint.transform.location
        return self._global_planner.trace_route(start_location, end_location)

    def done(self):
        """Check whether the agent has reached its destination."""
        return self._local_planner.done()

    def ignore_traffic_lights(self, active=True):
        """(De)activates the checks for traffic lights"""
        self._ignore_traffic_lights = active

    def ignore_stop_signs(self, active=True):
        """(De)activates the checks for stop signs"""
        self._ignore_stop_signs = active

    def ignore_vehicles(self, active=True):
        """(De)activates the checks for vehicles"""
        self._ignore_vehicles = active
    
    def activate_dlc(self, active=True):
        """(De)activate DLC"""
        if active:
            self._deactivate_dlc = False
        else:
            self._deactivate_dlc = True

    def lane_change(self, direction, same_lane_time=0, other_lane_time=0, lane_change_time=2):
        """
        Changes the path so that the vehicle performs a lane change.
        Use 'direction' to specify either a 'left' or 'right' lane change,
        and the other 3 fine tune the maneuver
        """
        speed = self._vehicle.get_velocity().length()
        path = self._generate_lane_change_path(
            self._map.get_waypoint(self._vehicle.get_location()),
            direction,
            same_lane_time * speed,
            other_lane_time * speed,
            lane_change_time * speed,
            False,
            1,
            self._sampling_resolution
        )
        if not path:
            print("WARNING: Ignoring the lane change as no path was found")

        self.set_global_plan(path)

    def _affected_by_traffic_light(self, lights_list=None, max_distance=None):
        """
        Method to check if there is a red light affecting the vehicle.

            :param lights_list (list of carla.TrafficLight): list containing TrafficLight objects.
                If None, all traffic lights in the scene are used
            :param max_distance (float): max distance for traffic lights to be considered relevant.
                If None, the base threshold value is used
        """
        if self._ignore_traffic_lights:
            return (False, None)

        if not lights_list:
            lights_list = self._world.get_actors().filter("*traffic_light*")

        if not max_distance:
            max_distance = self._base_tlight_threshold

        if self._last_traffic_light:
            if self._last_traffic_light.state != carla.TrafficLightState.Red:
                self._last_traffic_light = None
            else:
                return (True, self._last_traffic_light)

        for traffic_light in lights_list:
            if traffic_light.id in self._lights_map:
                trigger_wp = self._lights_map[traffic_light.id]
            else:
                trigger_location = get_trafficlight_trigger_location(traffic_light)
                trigger_wp = self._map.get_waypoint(trigger_location)
                self._lights_map[traffic_light.id] = trigger_wp

            if trigger_wp.transform.location.distance(self._vehicle_wp.location) > max_distance:
                continue

            if trigger_wp.road_id != self._vehicle_wp.road_id:
                continue

            ve_dir = self._vehicle_wp.transform.get_forward_vector()
            wp_dir = trigger_wp.transform.get_forward_vector()
            dot_ve_wp = ve_dir.x * wp_dir.x + ve_dir.y * wp_dir.y + ve_dir.z * wp_dir.z

            if dot_ve_wp < 0:
                continue

            if traffic_light.state != carla.TrafficLightState.Red:
                continue

            if is_within_distance(trigger_wp.transform, self._vehicle.get_transform(), max_distance, [0, 90]):
                self._last_traffic_light = traffic_light
                return (True, traffic_light)

        return (False, None)
    
    def _walker_obstacle_detected(self, walker_list, max_distance=None, up_angle_th=90, low_angle_th=0, lane_offset=0):
        """
        Method to check if there is a vehicle in the influencing range of the agent interupting its decision and movement.

            :param walker_list (list of carla.warker): list contatining carla.walker objects.
            :param max_distance: max freespace to check for obstacles.
                If None, the base threshold value is used
            :return (bool, carla.walker, distance from point to point)
        """
        def get_route_polygon():
            route_bb = []
            extent_y = self._vehicle.bounding_box.extent.y
            r_ext = extent_y + self._offset
            l_ext = -extent_y + self._offset
            r_vec = ego_transform.get_right_vector()
            p1 = ego_location + carla.Location(r_ext * r_vec.x, r_ext * r_vec.y)
            p2 = ego_location + carla.Location(l_ext * r_vec.x, l_ext * r_vec.y)
            route_bb.extend([[p1.x, p1.y, p1.z], [p2.x, p2.y, p2.z]])

            for wp, _ in self._local_planner.get_plan():
                if ego_location.distance(wp.transform.location) > max_distance:
                    break

                r_vec = wp.transform.get_right_vector()
                p1 = wp.transform.location + carla.Location(r_ext * r_vec.x, r_ext * r_vec.y)
                p2 = wp.transform.location + carla.Location(l_ext * r_vec.x, l_ext * r_vec.y)
                route_bb.extend([[p1.x, p1.y, p1.z], [p2.x, p2.y, p2.z]])

            # Two points don't create a polygon, nothing to check
            if len(route_bb) < 3:
                return None

            return Polygon(route_bb)

        if self._ignore_vehicles:
            return (False, None, -1)

        if not max_distance:
            max_distance = self._base_vehicle_threshold

        ego_transform = self._vehicle.get_transform()
        ego_location = ego_transform.location
        ego_wpt = self._map.get_waypoint(ego_location)

        # Get the right offset
        if ego_wpt.lane_id < 0 and lane_offset != 0:
            lane_offset *= -1

        opposite_invasion = abs(self._offset) + self._vehicle.bounding_box.extent.y > ego_wpt.lane_width / 2
        use_bbs = self._use_bbs_detection or opposite_invasion or ego_wpt.is_junction

        # Get the route bounding box
        route_polygon = get_route_polygon()

        obstacle_list = []
        for target_walker in walker_list:

            if target_walker.id == self._vehicle.id:
                continue

            target_transform = target_walker.get_transform()
            if target_transform.location.distance(ego_location) > max_distance:
                continue

            target_wpt = self._map.get_waypoint(target_transform.location, lane_type=carla.LaneType.Any)

            # General approach for junctions and vehicles invading other lanes due to the offset
            if (use_bbs or target_wpt.is_junction) and route_polygon:

                target_bb = target_walker.bounding_box
                target_vertices = target_bb.get_world_vertices(target_walker.get_transform())
                target_list = [[v.x, v.y, v.z] for v in target_vertices]
                target_polygon = Polygon(target_list)

                if route_polygon.intersects(target_polygon): # It lacks prediction of the target vehicle route
                    obstacle_list.append((True, target_walker, compute_distance(target_walker.get_location(), ego_location)))

            # Simplified approach, using only the plan waypoints (similar to TM)
            else:

                if target_wpt.lane_id != ego_wpt.lane_id  + lane_offset:
                    next_wpt = self._local_planner.get_incoming_waypoint_and_direction(steps=3)[0]
                    if not next_wpt:
                        continue
                    if target_wpt.lane_id != next_wpt.lane_id  + lane_offset:
                        continue

                if is_within_distance(target_transform, ego_transform, max_distance, [low_angle_th, up_angle_th]):
                    obstacle_list.append((True, target_walker, compute_distance(target_transform.location, ego_transform.location)))
        
        if obstacle_list:
            dis = [x[-1] for x in obstacle_list]
            min_dis = min(dis)
            dis_index = dis.index(min_dis)
            return obstacle_list[dis_index]
        else:
            return (False, None, -1)

    def _vehicle_obstacle_detected(self, vehicle_list=None, max_distance=None, up_angle_th=90, low_angle_th=0, lane_offset=0):
        """
        Method to check if there is a vehicle in the influencing range of the agent interupting its decision and movement.
        It remains a chanllange to detect relevant vehciles with decision and trajectory prediction technologies

            :param vehicle_list (list of carla.Vehicle): list contatining intelligent vehicle objects.
                If None, all vehicle in the scene are used
            :param max_distance: max freespace to check for obstacles.
                If None, the base threshold value is used
            :param [low_angle_th,up_angle_th]: angle range to detect a vehicle
            :param lane_offset: lane offset to determine target lane, but it is not the only lane to be detected
            :return (bool, carla.Vehicle, distance from point to point)
        """
        def get_route_polygon():
            route_bb = []
            extent_y = self._vehicle.bounding_box.extent.y
            r_ext = extent_y + self._offset
            l_ext = -extent_y + self._offset
            r_vec = ego_transform.get_right_vector()
            p1 = ego_location + carla.Location(r_ext * r_vec.x, r_ext * r_vec.y)
            p2 = ego_location + carla.Location(l_ext * r_vec.x, l_ext * r_vec.y)
            route_bb.extend([[p1.x, p1.y, p1.z], [p2.x, p2.y, p2.z]])

            for wp, _ in self._local_planner.get_plan():
                if ego_location.distance(wp.transform.location) > max_distance:
                    break

                r_vec = wp.transform.get_right_vector()
                p1 = wp.transform.location + carla.Location(r_ext * r_vec.x, r_ext * r_vec.y)
                p2 = wp.transform.location + carla.Location(l_ext * r_vec.x, l_ext * r_vec.y)
                route_bb.extend([[p1.x, p1.y, p1.z], [p2.x, p2.y, p2.z]])

            # Two points don't create a polygon, nothing to check
            if len(route_bb) < 3:
                return None

            return Polygon(route_bb)

        if self._ignore_vehicles:
            return (False, None, -1)

        if not vehicle_list:
            vehicle_list = self._world.get_actors().filter("*vehicle*")

        if not max_distance:
            max_distance = self._base_vehicle_threshold

        ego_transform = self._vehicle.get_transform()
        ego_location = ego_transform.location
        ego_wpt = self._map.get_waypoint(ego_location)

        # Get the right offset
        if ego_wpt.lane_id < 0 and lane_offset != 0:
            lane_offset *= -1

        opposite_invasion = abs(self._offset) + self._vehicle.bounding_box.extent.y > ego_wpt.lane_width / 2
        use_bbs = self._use_bbs_detection or opposite_invasion or ego_wpt.is_junction

        # Get the route bounding box
        route_polygon = get_route_polygon()

        obstacle_list = []
        for target_vehicle in vehicle_list:

            if target_vehicle.id == self._vehicle.id:
                continue

            target_transform = target_vehicle.get_transform()
            if target_transform.location.distance(ego_location) > max_distance:
                continue

            target_wpt = self._map.get_waypoint(target_transform.location, lane_type=carla.LaneType.Any)

            # General approach for junctions and vehicles invading other lanes due to the offset
            if (use_bbs or target_wpt.is_junction) and route_polygon and not self._deactivate_bbs_detection:

                target_bb = target_vehicle.bounding_box
                target_vertices = target_bb.get_world_vertices(target_vehicle.get_transform())
                target_list = [[v.x, v.y, v.z] for v in target_vertices]
                target_polygon = Polygon(target_list)

                if route_polygon.intersects(target_polygon): # It lacks prediction of the target vehicle route
                    obstacle_list.append((True, target_vehicle, compute_distance(target_vehicle.get_location(), ego_location)))     

            # Simplified approach, using only the plan waypoints (similar to TM)
            else:

                if target_wpt.lane_id == -1 and target_wpt.transform.location.y > 9:
                    target_lane_id = -4
                else:
                    target_lane_id = target_wpt.lane_id
                
                if target_lane_id != ego_wpt.lane_id  + lane_offset:
                    continue
                    # next_wpt = self._local_planner.get_incoming_waypoint_and_direction(steps=3)[0]
                    # if not next_wpt:
                    #     continue
                    # if target_lane_id != next_wpt.lane_id  + lane_offset:
                    #     continue

                if is_within_distance(target_transform, ego_transform, max_distance, [low_angle_th, up_angle_th]):
                    obstacle_list.append((True, target_vehicle, compute_distance(target_transform.location, ego_transform.location)))
        
        if obstacle_list:
            dis = [x[-1] for x in obstacle_list]
            min_dis = min(dis)
            dis_index = dis.index(min_dis)
            return obstacle_list[dis_index]
        else:
            return (False, None, -1)

    def _generate_lane_change_path(self, waypoint, direction='left', distance_same_lane=10, # The ability of lane change is limited, 
                                distance_other_lane=25, lane_change_distance=25, # if the next point is not allowed for a lane change
                                check=True, lane_changes=1, step_distance=2): # the maneuver cannot be conducted. A search process is necessary
        """
        This methods generates a path that results in a lane change.
        Use the different distances to fine-tune the maneuver.
        If the lane change is impossible, the returned path will be empty.
        """
        distance_same_lane = max(distance_same_lane, 0.1)
        distance_other_lane = max(distance_other_lane, 0.1)
        lane_change_distance = max(lane_change_distance, 0.1)

        plan = []
        plan.append((waypoint, RoadOption.LANEFOLLOW))  # start position

        option = RoadOption.LANEFOLLOW

        # Same lane
        distance = 0
        while distance < distance_same_lane:
            next_wps = plan[-1][0].next(step_distance)
            if not next_wps:
                return []
            next_wp = next_wps[0]
            distance += next_wp.transform.location.distance(plan[-1][0].transform.location)
            plan.append((next_wp, RoadOption.LANEFOLLOW))

        if direction == 'left':
            option = RoadOption.CHANGELANELEFT
        elif direction == 'right':
            option = RoadOption.CHANGELANERIGHT
        else:
            # ERROR, input value for change must be 'left' or 'right'
            return []

        lane_changes_done = 0
        lane_change_distance = lane_change_distance / lane_changes

        # Lane change
        while lane_changes_done < lane_changes:

            # Move forward
            next_wps = plan[-1][0].next(lane_change_distance)
            if not next_wps:
                return []
            next_wp = next_wps[0]

            # Get the side lane
            if direction == 'left':
                if check and str(next_wp.lane_change) not in ['Left', 'Both']:
                    return []
                side_wp = next_wp.get_left_lane()
            else:
                if check and str(next_wp.lane_change) not in ['Right', 'Both']:
                    return []
                side_wp = next_wp.get_right_lane()

            if not side_wp or side_wp.lane_type != carla.LaneType.Driving:
                return []

            # Update the plan
            plan.append((side_wp, option))
            lane_changes_done += 1

        # Other lane
        distance = 0
        while distance < distance_other_lane:
            next_wps = plan[-1][0].next(step_distance)
            if not next_wps:
                return []
            next_wp = next_wps[0]
            distance += next_wp.transform.location.distance(plan[-1][0].transform.location)
            plan.append((next_wp, RoadOption.LANEFOLLOW))

        return plan
    
    def _update_information(self,current_time):
        """
        This method updates the information regarding the ego vehicle based on the surrounding world and
        updates information of all the vehciles to vehicle register.
            :param current_time: current time in the simulation from the begining
        """
        self._speed = get_speed(self._vehicle) # km/h
        location = self._vehicle.get_location()
        self._vehicle_wp = self._map.get_waypoint(location)
        self._lane_id = self._vehicle_wp.lane_id

        # determine lane change parameters
        if self._off_ramp:
            if location.x < self._off_ramp_influence_range_x[0] or location.x > self._off_ramp_influence_range_x[1]:
                self._safe_acc_ratio = 1
                self._lambda_factor = 0.4
            else:
                factor = (location.x - self._off_ramp_influence_range_x[0]) / (self._off_ramp_influence_range_x[1]-self._off_ramp_influence_range_x[0])
                self._safe_acc_ratio = 1
                self._lambda_factor = factor * 1 + (1-factor) * 0.4
            if self._vehicle_type == 'cav':
                self._safe_acc = - self._safe_acc_ratio * 4
            else:
                self._safe_acc = - self._safe_acc_ratio * 4

        # check lane change manuver based on changes of lane id
        if self._lane_id != self._last_lane_id:
            self._lane_change_flag = False
            self._lane_change_direction = 0
            if self._remaining_lane_change:
                del self._remaining_lane_change[0]
        
        # get speed limit from the map through client or function get_speed_limit() defined for a specific map
        self._speed_limit = get_speed_limit(self._vehicle_wp) if self._activate_specific_map_speed_limit else self._vehicle.get_speed_limit()
        lane_change_list = self._local_planner.update_waypoint_queue()
        if lane_change_list:
            for lane_change in lane_change_list:
                self._remaining_lane_change.append(lane_change)
        target_speed = min([self._behavior.max_speed, self._speed_limit - self._behavior.speed_lim_dist])
        self._target_speed = target_speed
        self._local_planner.set_speed(target_speed)

        # fix chanllange in determing decision of whether chane lane or turn in intersection
        self._direction = self.get_direction()
        if self._direction is None:
            self._direction = RoadOption.LANEFOLLOW
        self._vehicle_register.upload_information(self._int_vehicle,current_time,self._lane_id)

        self._look_ahead_steps = int((self._speed_limit) / 10) # The determination of the look ahead steps needs carefully considered
        self._incoming_waypoint, self._incoming_direction = self._local_planner.get_incoming_waypoint_and_direction(
            steps=self._look_ahead_steps)
        if self._incoming_direction is None:
            self._incoming_direction = RoadOption.LANEFOLLOW
        
        # if self._behavior.tailgate_counter > 0:
        #     self._behavior.tailgate_counter -= 1
        
        if self._behavior.lane_change_counter > 0:
            self._behavior.lane_change_counter -= 1
    
    def get_time_gap(self, target_vehicle):
        """
        This function determines desired time gap based on vehicle type of the target vehicle
            :param target_vehicle: carla.Vehicle
        """
        if self._vehicle_type == 'cav':
            time_gap = self._behavior.time_gap_acc
            if target_vehicle:
                agent = self._vehicle_register.id_agent[target_vehicle.id]
                if agent:
                    if agent._vehicle_type == 'cav':
                        time_gap = self._behavior.time_gap_cacc
        else:
            time_gap = self._behavior.time_gap_hdv
        return time_gap
    
    def get_direction(self):
        """
        There is a problem in determing the driving decisions in the intersection area.
        There needs an more effective method to determine it is LaneFollow or LaneChange in intersection
        This function develops a simple method by changing LaneFollow decision to LaneChange decision based 
        on waypoints deleted by the local planner, but it still don't work in some situations
        """
        direction = self._local_planner.target_road_option
        if direction not in [RoadOption.CHANGELANERIGHT, RoadOption.CHANGELANELEFT] and self._remaining_lane_change:
            if self._remaining_lane_change[0] == 'right':
                direction = RoadOption.CHANGELANERIGHT
            elif self._remaining_lane_change[0] == 'left':
                direction = RoadOption.CHANGELANELEFT
        return direction

    def traffic_light_manager(self):
        """
        This method is in charge of behaviors for red lights.
        """
        actor_list = self._world.get_actors()
        lights_list = actor_list.filter("*traffic_light*")
        affected, _ = self._affected_by_traffic_light(lights_list)

        return affected

    def _tailgating(self, waypoint, vehicle_list):
        """
        This method is in charge of tailgating behaviors.

            :param waypoint: current waypoint of the agent
            :param vehicle_list: list of all the nearby vehicles
        """

        left_turn = waypoint.left_lane_marking.lane_change
        right_turn = waypoint.right_lane_marking.lane_change

        left_wpt = waypoint.get_left_lane()
        right_wpt = waypoint.get_right_lane()

        behind_vehicle_state, behind_vehicle, _ = self._vehicle_obstacle_detected(vehicle_list, max(
            self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=180, low_angle_th=90)
        if behind_vehicle_state and self._speed < get_speed(behind_vehicle):
            if (right_turn == carla.LaneChange.Right or right_turn ==
                    carla.LaneChange.Both) and waypoint.lane_id * right_wpt.lane_id > 0 and right_wpt.lane_type == carla.LaneType.Driving:
                new_vehicle_state, _, _ = self._vehicle_obstacle_detected(vehicle_list, max(
                    self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=180, lane_offset=1)
                if not new_vehicle_state:
                    print("Tailgating, moving to the right!")
                    end_waypoint = self._local_planner._waypoints_queue[-1][0]
                    self._behavior.tailgate_counter = 200
                    self.set_destination(end_waypoint.transform.location,
                                         right_wpt.transform.location)
            elif left_turn == carla.LaneChange.Left and waypoint.lane_id * left_wpt.lane_id > 0 and left_wpt.lane_type == carla.LaneType.Driving:
                new_vehicle_state, _, _ = self._vehicle_obstacle_detected(vehicle_list, max(
                    self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=180, lane_offset=-1)
                if not new_vehicle_state:
                    print("Tailgating, moving to the left!")
                    end_waypoint = self._local_planner._waypoints_queue[-1][0]
                    self._behavior.tailgate_counter = 200
                    self.set_destination(end_waypoint.transform.location,
                                         left_wpt.transform.location)

    def collision_and_car_avoid_manager(self, max_distance, vehicle_list):
        """
        This module is in charge of warning in case of a collision, but it is not effective in some situations

            :param max_distance: max distance to detedt a vehicle
            :param vehicle_list: list of vehicles to be detected
            :return vehicle_state: True if there is a vehicle nearby, False if not
            :return vehicle: nearby vehicle
            :return distance: distance to nearby vehicle from center to center
        """

        if self._direction == RoadOption.CHANGELANELEFT:
            vehicle_state, vehicle, distance = self._vehicle_obstacle_detected(
                vehicle_list, max_distance, up_angle_th=180, lane_offset=-1)
        elif self._direction == RoadOption.CHANGELANERIGHT:
            vehicle_state, vehicle, distance = self._vehicle_obstacle_detected(
                vehicle_list, max_distance, up_angle_th=180, lane_offset=1)
        else:
            vehicle_state, vehicle, distance = self._vehicle_obstacle_detected(
                vehicle_list, max_distance, up_angle_th=90)

        return vehicle_state, vehicle, distance

    def pedestrian_avoid_manager(self, waypoint):
        """
        This module is in charge of warning in case of a collision
        with any pedestrian.

            :param location: current location of the agent
            :param waypoint: current waypoint of the agent
            :return vehicle_state: True if there is a walker nearby, False if not
            :return vehicle: nearby walker
            :return distance: distance to nearby walker
        """

        walker_list = self._world.get_actors().filter("*walker.pedestrian*")
        def dist(w): return w.get_location().distance(waypoint.transform.location)
        walker_list = [w for w in walker_list if dist(w) < 10]

        if self._direction == RoadOption.CHANGELANELEFT:
            walker_state, walker, distance = self._walker_obstacle_detected(walker_list, max(
                self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=90, lane_offset=-1)
        elif self._direction == RoadOption.CHANGELANERIGHT:
            walker_state, walker, distance = self._walker_obstacle_detected(walker_list, max(
                self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=90, lane_offset=1)
        else:
            walker_state, walker, distance = self._walker_obstacle_detected(walker_list, max(
                self._behavior.min_proximity_threshold, self._speed_limit / 3), up_angle_th=60)

        return walker_state, walker, distance
    
    def acceleration_calculation_IDM(self, leader, follower):
        """
        This function caluculates acceleration of the follower based on IDM car-following model
            :param leader: preceding vehicle
            :param follower: vehicle needs calculation
            :return output: acceleration based on IDM
        """

        a = self._local_planner._vehicle_controller._lon_controller.a
        b = self._local_planner._vehicle_controller._lon_controller.b
        target_vel = self._local_planner._target_speed / 3.6
        ego_vel = get_speed(follower) / 3.6
        if leader:
            target_vehicle_vel = get_speed(leader) / 3.6

            ego_transform = follower.get_transform()
            ego_front_transform = ego_transform
            ego_front_transform.location += carla.Location(
                follower.bounding_box.extent.x * ego_transform.get_forward_vector())
            
            target_transform = leader.get_transform()
            target_rear_transform = target_transform
            target_rear_transform.location -= carla.Location(
                leader.bounding_box.extent.x * target_transform.get_forward_vector())
            
            distance = compute_distance(target_rear_transform.location, ego_front_transform.location)
            time_gap = self.get_time_gap(None)
            s_star = max(time_gap * ego_vel + ego_vel*(ego_vel-target_vehicle_vel)/(2*(a*b)**0.5),0) + self._behavior.stand_still_dis
            output = a * (1 - (ego_vel/target_vel)**4 - (s_star/distance)**2)
        else:
            output = a * (1 - (ego_vel/target_vel)**4)

        return output
    
    def mobil(self,turn_wp,preceding_vehicle,front_vehicle,back_vehicle,max_distance,vehicle_list,politeness_factor=0.5,lane_change_threshold=1):
        """
        This function builds a MOBIL car-following model
        reference: General Lane-Changing Model MOBIL for Car-Following Models (Kesting et al., 2007)
            :param turn_wp: waypoint to change lane to on the target lane
            :param preceding_vehicle: preceding vehicle on the current lane
            :param front_vehicle: front vehicle on the target lane
            :param back_vehicle: back vehicle on the target lane
            :param max_distance: max distance to detect a vehicle
            :param vehicle_list: list of vehicles detected
            :return politeness_factor: politeness factor in MOBIL
            :return lane_change_threshold: lane change threshold in MOBIL
        """

        safe_acceleration = self._safe_acc
        acc_sub_bef = self.acceleration_calculation_IDM(preceding_vehicle, self._vehicle)
        acc_sub_aft = self.acceleration_calculation_IDM(front_vehicle, self._vehicle)

        # get current follower
        follower_state, follower_vehicle, _ = self._vehicle_obstacle_detected(vehicle_list, 
                                                                max_distance, up_angle_th=180, low_angle_th=90, lane_offset=0)
        if follower_state:
            acc_old_fol_bef = self.acceleration_calculation_IDM(self._vehicle, follower_vehicle)
            acc_old_fol_aft = self.acceleration_calculation_IDM(preceding_vehicle, follower_vehicle)
        else:
            acc_old_fol_bef = 0
            acc_old_fol_aft = 0
        
        # get follower on the target lane
        if back_vehicle:
            acc_new_fol_bef = self.acceleration_calculation_IDM(front_vehicle, back_vehicle)
            acc_new_fol_aft = self.acceleration_calculation_IDM(self._vehicle, back_vehicle)
        else:
            acc_new_fol_bef = 0
            acc_new_fol_aft = 0
        
        if acc_new_fol_aft <= safe_acceleration or acc_sub_aft <= safe_acceleration:
            return (False, None)
        
        loss = acc_sub_aft - acc_sub_bef + politeness_factor * \
        (acc_old_fol_aft - acc_old_fol_bef + acc_new_fol_aft - acc_new_fol_bef)

        if loss >= lane_change_threshold and self._behavior.lane_change_counter == 0:
            lane_change_flag = True
            lane_change_wp = turn_wp
        else:
            lane_change_flag = False
            lane_change_wp = None
        
        return (lane_change_flag, lane_change_wp)
    
    def discretionary_lane_change_judgement(self, max_distance, preceding_vehicle, preceding_distance, vehicle_list):
        """
        Module judging whether a discretionary lane change can be conducted

            :param max_distance: max_distance to detect vehicle
            :param preceding_vehicle: preceding vehicle on the current lane
            :param preceding_distance: distance between the front vehicle and the ego vehicle
            :param vehicle_list: list of vehicle in the influence range
            :return lane_change_flag: whether to change lane (bool)
            :return lane_change_wp: the lane change waypoint
            :return front_vehicle: the front vehicle on the target lane if it exists
            :return change_direction: 0-follow the lane, 1-right, -1-left
        """

        delta_vel_acc = 1
        t_lane_change =  1
        lambda_factor = self._lambda_factor
        vehicle_vel = self._speed / 3.6 # m/s
        
        max_vehicle_vel = min([self._behavior.max_speed, self._speed_limit - self._behavior.speed_lim_dist]) / 3.6

        left_turn = self._vehicle_wp.left_lane_marking.lane_change
        right_turn = self._vehicle_wp.right_lane_marking.lane_change

        left_wpt = self._vehicle_wp.get_left_lane()
        right_wpt = self._vehicle_wp.get_right_lane()

        def judge_one_side(turn_wp, turn_dir, turn_permit, lane_offset):
            """ check whether the ego vehicle can change the lane on one side """

            lane_change_wp = None
            lane_change_flag = False
            front_vehicle = None
            
            if turn_wp:
                # if destination of the ego vehicle is not off-ramp, it can not change lane to lane -4
                if not self._off_ramp and turn_wp.lane_id == -4:
                    return (lane_change_flag, lane_change_wp, front_vehicle)
            
                # check is it permited to change lane physically
                if (turn_dir == turn_permit or turn_dir == carla.LaneChange.Both) and dcavl_permit(turn_wp, self._vehicle_type) \
                    and self._vehicle_wp.lane_id * turn_wp.lane_id > 0 and turn_wp.lane_type == carla.LaneType.Driving:

                    front_state, front_vehicle, front_distance = self._vehicle_obstacle_detected(vehicle_list, 
                                                                                max_distance, up_angle_th=90, lane_offset=lane_offset)
                    back_state, back_vehicle, back_distance = self._vehicle_obstacle_detected(vehicle_list, 
                                                                                max_distance, up_angle_th=180, low_angle_th=90, lane_offset=lane_offset)
                    state, _, distance = self._vehicle_obstacle_detected(vehicle_list, 
                                                                                max_distance, up_angle_th=180, lane_offset=lane_offset)
                    
                    # check is there a changing chance considering vehicles on the target lane
                    if self._vehicle_type == 'cav':
                        # reference: Route Control Strategies for Autonomous Vehicles Existing to Off-Ramps (Dong et al., 2020)
                        # flag_front = not front_state or (front_distance > preceding_distance + 5)
                        flag_front = self.check_lane_change_condition_in_the_front(preceding_vehicle,front_vehicle)
                        if back_state:
                            back_vehicle_vel = get_speed(back_vehicle) / 3.6
                            safe_distance = lambda_factor*(back_vehicle_vel-self.cal_vehicle_forward_vel())*t_lane_change + (1-lambda_factor)*max_vehicle_vel*t_lane_change
                            flag_back = back_distance > safe_distance + 5
                        else:
                            flag_back = True
                        
                        time_gap = self.get_time_gap(preceding_vehicle)
                        lane_follow_distance = self._base_vehicle_threshold + time_gap * vehicle_vel
                        if flag_front and flag_back and vehicle_vel<max_vehicle_vel and self._behavior.lane_change_counter == 0 and \
                            preceding_distance<lane_follow_distance: # min(vehicle_vel+delta_vel_acc,max_vehicle_vel)*t_lane_change
                            lane_change_flag = True
                            lane_change_wp = turn_wp
                        
                    else:
                        # reference: General Lane-Changing Model MOBIL for Car-Following Models (Kesting et al., 2007)
                        politeness_factor = self._behavior.politeness_factor
                        lane_change_threshold = self._behavior.lane_change_threshold
                        lane_change_flag, lane_change_wp = self.mobil(turn_wp,preceding_vehicle,front_vehicle,back_vehicle,max_distance,vehicle_list,
                                                                      politeness_factor,lane_change_threshold)
                    
                    # vehicle on the target lane is too near
                    if state:
                        if distance <= 10:
                            lane_change_wp = None
                            lane_change_flag = False
                            front_vehicle = None
                    
                    lane_change_flag = self.check_other_lane_change_manevers(lane_change_flag,lane_change_wp,vehicle_list)
            
            return (lane_change_flag, lane_change_wp, front_vehicle)
        
        left_lane_change_flag, left_lane_change_wp, left_front_vehicle = judge_one_side(left_wpt, left_turn, carla.LaneChange.Left, -1)
        right_lane_change_flag, right_lane_change_wp, right_front_vehicle = judge_one_side(right_wpt, right_turn, carla.LaneChange.Right, 1)

        change_direction = 0
        if left_lane_change_flag and right_lane_change_flag:
            choice = random.random()
            if choice < 0.5:
                lane_change_flag = left_lane_change_flag
                lane_change_wp = left_lane_change_wp
                front_vehicle = left_front_vehicle
                change_direction = -1
            else:
                lane_change_flag = right_lane_change_flag
                lane_change_wp = right_lane_change_wp
                front_vehicle = right_front_vehicle
                change_direction = 1
        elif left_lane_change_flag and not right_lane_change_flag:
            lane_change_flag = left_lane_change_flag
            lane_change_wp = left_lane_change_wp
            front_vehicle = left_front_vehicle
            change_direction = -1
        elif not left_lane_change_flag and right_lane_change_flag:
            lane_change_flag = right_lane_change_flag
            lane_change_wp = right_lane_change_wp
            front_vehicle = right_front_vehicle
            change_direction = 1
        else:
            lane_change_flag = False
            lane_change_wp = None
            front_vehicle = None
        
        return (lane_change_flag, lane_change_wp, front_vehicle, change_direction)

    
    def discretionary_lane_change(self, max_distance, preceding_vehicle, preceding_distance, vehicle_list, debug=False):
        """
        Module in charge of discretionary lane change and car-following behaviors. The vehicle can follow a preceding vehicle if
        it plans to keep the lane. Besides, if the vehicle front blocking the way and there are chances to move faster on
        the other lane, the ego vehicle will change lane automaticly.

            :param max_distance: max_distance to detect vehicle
            :param preceding_distance: distance between the front vehicle and the ego vehicle
            :param vehicle_list: list of vehicles in the influence range
            :param debug: boolean for debugging
            :return control, transform, velocity to be applied
        """
        target_vehicle = preceding_vehicle
        time_gap = self.get_time_gap(target_vehicle)

        if preceding_vehicle and not self._lane_change_flag:

            lane_change_flag, lane_change_wp, front_vehicle, change_dir = self.discretionary_lane_change_judgement(max_distance, preceding_vehicle,
                                                                                                        preceding_distance, vehicle_list)
            # deactivate DLC module
            if self._deactivate_dlc:
                lane_change_flag = False
            
            if self._activate_platoon_control and self.is_within_platoon():
                lane_change_flag = False
            
            if lane_change_flag:
                self._target_lane_id = lane_change_wp.lane_id
                print("Discretionary lane change! vehicle type:", self._vehicle_type)
                end_transform = self._local_planner._waypoints_queue[-1][0].transform
                if self._available_destination:
                    if self._off_ramp and not self._change_destination:
                        end_transform = self._local_planner._waypoints_queue[-1][0].transform
                    elif self._lane_id == -1:
                        end_transform = self._available_destination[0]
                    elif self._lane_id == -2:
                        end_transform = self._available_destination[1]
                    elif self._lane_id == -3:
                        end_transform = self._available_destination[2]
                self._behavior.lane_change_counter = 100 # lane change time threshold (lane_change_counter * delta)
                self.set_destination(end_transform.location, lane_change_wp.transform.location)                  
                self._lane_change_flag = True
                self._lane_change_direction = change_dir

        if self._lane_change_flag:
            _, front_vehicle, _ = self._vehicle_obstacle_detected(vehicle_list, 
                                                        max_distance, up_angle_th=90, lane_offset=self._lane_change_direction)
            target_vehicle = front_vehicle
            time_gap = self.get_time_gap(target_vehicle)
            
        control, transform, velocity = self._local_planner.run_step(target_vehicle, time_gap, debug)
        self._target_vehicle = target_vehicle
        
        # platoon control
        control, transform, velocity = self.platoon_control(control, transform, velocity, target_vehicle)

        # lane change while following the preceding vehicle in current lane to avoid crash
        time_gap = self.get_time_gap(preceding_vehicle)
        control, transform, velocity = self.follow_current_lane_when_change_lane(control, transform, velocity, preceding_vehicle, time_gap)

        return (control, transform, velocity)
    
    def decelerate(self, delta_speed=10):
        """
        Decelerate to find possible gap to change lane
            :param delta_speed: gap from speed limit when decelerating (km/h)
        """
        target_speed = max(min([self._behavior.max_speed, self._speed_limit - self._behavior.speed_lim_dist]) - delta_speed, 0)
        self._target_speed = target_speed
        self._local_planner.set_speed(target_speed)
    
    def go_to_next_destination(self,preceding_vehicle):
        """
        This function change the destination of the vehicle if it originally plans to 
        change lane to off-ramp but fail before reaching end of the freeway. This vehicle
        has to go ahead and finds next off-ramp
            :param preceding vehicle to follow
        """

        location = self._vehicle.get_location()
        if self._lane_id == -1 and location.x>0 and location.y>9:
            lane_id = -4
        else:
            lane_id = self._lane_id
        
        if lane_id == -1:
            destination = self._available_destination[0]
        elif lane_id == -2:
            destination = self._available_destination[1]
        elif lane_id == -3:
            destination = self._available_destination[2]
        else:
            destination = self._available_destination[3]
        self.set_destination(destination.location,location,True)
        time_gap = self.get_time_gap(preceding_vehicle)
        control, transform, velocity = self._local_planner.run_step(preceding_vehicle, time_gap)
        if lane_id != -4 and self._off_ramp:
            self._change_destination = True
            print(f'There is no chance to change lane, go ahead. vehicle type: {self._vehicle_type}, id: {self._vehicle.id}')

        return (control, transform, velocity)
    
    def mandatory_lane_change(self, max_distance, vehicle_list, debug=False):
        """
        Module in charge of mandatory lane change and car-following behaviors. It can check the lane change condition, 
        follow the front vehicle on the target lane, and brake to let nearby vehicle pass if needed.

            :param max_distance: max_distance to detect vehicle
            :param vehicle_list: list of vehicle in the influence range
            :param debug: boolean for debugging
            :return control: carla.VehicleControl 
        """

        lambda_factor = self._lambda_factor
        t_lane_change = 1
        max_vehicle_vel = self._target_speed / 3.6

        left_wpt = self._vehicle_wp.get_left_lane()
        right_wpt = self._vehicle_wp.get_right_lane()

        # get preceding vehicle information on the same lane
        _, preceding_vehicle, _ = self._vehicle_obstacle_detected(vehicle_list, 
                                                                        max_distance, up_angle_th=90, lane_offset=0)
        target_vehicle = preceding_vehicle
        time_gap = self.get_time_gap(target_vehicle)

        # check is it needed to change lane based on the local planner
        lane_change_wp = None
        
        if self._direction == RoadOption.CHANGELANERIGHT: # what if lane change mark forbids a lane change if self._direction == RoadOption.CHANGELANERIGHT
            front_state, front_vehicle, front_distance = self._vehicle_obstacle_detected(vehicle_list, 
                                                                        max_distance, up_angle_th=90, lane_offset=1)
            back_state, back_vehicle, back_distance = self._vehicle_obstacle_detected(vehicle_list, 
                                                                        max_distance, up_angle_th=180, low_angle_th=90, lane_offset=1)
            state, _, distance = self._vehicle_obstacle_detected(vehicle_list, 
                                                                        max_distance, up_angle_th=180, lane_offset=1)
            lane_change_wp = right_wpt
        elif self._direction == RoadOption.CHANGELANELEFT:
            front_state, front_vehicle, front_distance = self._vehicle_obstacle_detected(vehicle_list, 
                                                                        max_distance, up_angle_th=90, lane_offset=-1)
            back_state, back_vehicle, back_distance = self._vehicle_obstacle_detected(vehicle_list, 
                                                                        max_distance, up_angle_th=180, low_angle_th=90, lane_offset=-1)
            state, _, distance = self._vehicle_obstacle_detected(vehicle_list, 
                                                                        max_distance, up_angle_th=180, lane_offset=-1)
            lane_change_wp = left_wpt
        
        # only flag_front and flag_back are True, vehicle can change the lane
        lane_change_flag = False
        lane_follow_distance = self._base_vehicle_threshold + time_gap * self._speed / 3.6
        flag_front = not front_state or front_distance > lane_follow_distance
        if back_state:
            back_vehicle_vel = get_speed(back_vehicle) / 3.6
            safe_distance = lambda_factor*(back_vehicle_vel-self.cal_vehicle_forward_vel())*t_lane_change + (1-lambda_factor)*max_vehicle_vel*t_lane_change
            flag_back = back_distance > safe_distance + 5
        else:
            flag_back = True
        
        if flag_front and flag_back:
            lane_change_flag = True
            
        # vehicle on the target lane is too near
        if state:
            if distance <= 10:
                lane_change_flag = False
        
        lane_change_flag = self.check_other_lane_change_manevers(lane_change_flag,lane_change_wp,vehicle_list)

        if lane_change_flag:
            if lane_change_wp:
                if not self._lane_change_flag:
                    self._target_lane_id = lane_change_wp.lane_id
                    print("Mandatory Lane change")                 
                    self._lane_change_flag = True

                if front_state:
                    target_vehicle = front_vehicle
                    time_gap = self.get_time_gap(target_vehicle)

                control, transform, velocity = self._local_planner.run_step(target_vehicle, time_gap, debug)
            else:
                # bug
                control, transform, velocity = self._local_planner.run_step(target_vehicle, time_gap, debug)
        else:
            if self._available_destination and self._activate_change_destination:
                control, transform, velocity = self.go_to_next_destination(target_vehicle)
            else:
                control, transform, velocity = self.emergency_stop()
        
        # platoon control
        if self._activate_platoon_control:
            platoon = self.is_within_platoon()
            if platoon:
                platoon.vehicle_diverge(self._vehicle)
        
        # lane change while following the preceding vehicle in current lane to avoid crash
        time_gap = self.get_time_gap(preceding_vehicle)
        control, transform, velocity = self.follow_current_lane_when_change_lane(control, transform, velocity, preceding_vehicle, time_gap)
        
        return (control, transform, velocity)
    
    def lane_change_in_off_ramp_judgement(self):
        """
        This module judges whether a vehicle is currently in the off ramp influence area and
        its destination is off ramp

            :return judgement: whether does it satisfy the conditions (bool)
        """

        judgement = False
        waypoint = self._local_planner._waypoints_queue[-1][0]
        if self._off_ramp and waypoint.transform.location.y > 9 and waypoint.lane_id == -1:
            vehicle_location = self._vehicle_wp.transform.location
            location_x = vehicle_location.x
            if location_x > self._off_ramp_influence_range_x[0] and location_x < self._off_ramp_influence_range_x[1]:
                judgement = True
        return judgement

    def lane_change_in_off_ramp(self, max_distance, vehicle_list, debug=False):
        """
        This module controls a vehicle planning to go off ramp change lane before reaching to the only
        lane change point calculated by the global planner
            :param max_distance: max distance to detect a vehicle
            :param vehicle_list: list of carla.Vehicle
        """
        _, preceding_vehicle, _ = self._vehicle_obstacle_detected(vehicle_list, max_distance, up_angle_th=90)

        # check is it on the off ramp lane (lane_id=-4)
        lane_change_flag = False
        target_vehicle = preceding_vehicle
        time_gap = self.get_time_gap(target_vehicle)

        right_turn = self._vehicle_wp.right_lane_marking.lane_change
        right_wpt = self._vehicle_wp.get_right_lane()

        if right_turn and right_wpt:

            # check is it permitted to change lane
            if (right_turn == carla.LaneChange.Right or right_turn == carla.LaneChange.Both) and dcavl_permit(right_wpt, self._vehicle_type) \
                and self._vehicle_wp.lane_id * right_wpt.lane_id > 0 and right_wpt.lane_type == carla.LaneType.Driving:
                lane_change_wp = right_wpt
                
                # check is there a changing chance, only flag_front and flag_back are True, vehicle can change the lane
                # self.decelerate(20)
                # lane_change_flag = self.check_lane_change_condition_in_off_ramp(vehicle_list,max_distance,lane_change_wp)
                if self._behavior.vehicle_type == 'cav':
                    lane_change_flag = self.search_gap_to_change_lane(vehicle_list,max_distance,lane_change_wp)
                else:
                    # decelerate to find possible gap
                    if self._activate_deceleration_off_ramp:
                        self.decelerate(20)
                    lane_change_flag = self.check_lane_change_condition_in_off_ramp(vehicle_list,max_distance,lane_change_wp)

        if lane_change_flag and not self._lane_change_flag:
            self._target_lane_id = lane_change_wp.lane_id
            print("Lane change in off ramp influence area. vehicle type:", self._vehicle_type)
            end_waypoint = self._local_planner._waypoints_queue[-1][0]
            self.set_destination(end_waypoint.transform.location,
                                    lane_change_wp.transform.location)                   
            self._lane_change_flag = True
            self._lane_change_direction = 1
        
        if self._lane_change_flag:
            _, front_vehicle, _ = self._vehicle_obstacle_detected(vehicle_list, 
                                    max_distance, up_angle_th=90, lane_offset=self._lane_change_direction)
            target_vehicle = front_vehicle # there remains a problem if vehcles both exist at preceding and front position
            time_gap = self.get_time_gap(target_vehicle)
            
        control, transform, velocity = self._local_planner.run_step(target_vehicle, time_gap, debug)
        self._target_vehicle = target_vehicle

        # platoon control
        control, transform, velocity = self.platoon_control(control, transform, velocity, target_vehicle)

        # lane change while following the preceding vehicle in current lane to avoid crash
        time_gap = self.get_time_gap(preceding_vehicle)
        control, transform, velocity = self.follow_current_lane_when_change_lane(control, transform, velocity, preceding_vehicle, time_gap)

        return (control, transform, velocity)
    
    def check_lane_change_condition_in_off_ramp(self,vehicle_list,max_distance,lane_change_wp):
        """ This module checks if a vehcile can change lane to off ramp"""
        lambda_factor = self._lambda_factor
        t_lane_change = 1
        max_vehicle_vel = self._target_speed / 3.6
        front_state, front_vehicle, front_distance = self._vehicle_obstacle_detected(vehicle_list, 
                                                                    max_distance, up_angle_th=90, lane_offset=1)
        back_state, back_vehicle, back_distance = self._vehicle_obstacle_detected(vehicle_list, 
                                                                    max_distance, up_angle_th=180, low_angle_th=90, lane_offset=1)
        state, _, distance = self._vehicle_obstacle_detected(vehicle_list, 
                                                                    max_distance, up_angle_th=180, lane_offset=1)
        lon_controller = self._local_planner._vehicle_controller._lon_controller
        front_time_gap = self.get_time_gap(front_vehicle)
        front_acc = lon_controller.run_step(self._target_speed, front_vehicle, front_time_gap)
        if self._vehicle_type == 'cav':
            flag_front = not front_state or front_acc > self._safe_acc
            if back_state:
                back_vehicle_vel = get_speed(back_vehicle) / 3.6
                safe_distance = lambda_factor*(back_vehicle_vel-self.cal_vehicle_forward_vel())*t_lane_change + (1-lambda_factor)*max_vehicle_vel*t_lane_change
                flag_back = back_distance > safe_distance + 5
            else:
                flag_back = True
        else:
            flag_front = not front_state or front_acc > self._safe_acc
            if back_state:
                acc_new_fol_aft = self.acceleration_calculation_IDM(self._vehicle, back_vehicle)
                flag_back = acc_new_fol_aft > self._safe_acc
            else:
                flag_back = True
        
        lane_change_flag = False
        if flag_front and flag_back:
            lane_change_flag = True
        
        # vehicle on the target lane is too near
        # if state:
        #     if distance <= 10:
        #         lane_change_flag = False
        
        # predict: get lane change decisions of other vehicles
        lane_change_flag = self.check_other_lane_change_manevers(lane_change_flag,lane_change_wp,vehicle_list)
        return lane_change_flag
    
    def search_gap_to_change_lane(self,vehicle_list,max_distance,lane_change_wp):
        """ This module searches optimal gap for lane change manuver in off-ramp influence area """
        ego_transform = self._vehicle.get_transform()
        ego_location = ego_transform.location

        # get vehicles and their locations on the target lane
        target_vehicle_list = []
        target_vehicle_x = []
        for target_vehicle in vehicle_list:
            if target_vehicle.id == self._vehicle.id:
                continue
            target_transform = target_vehicle.get_transform()
            if target_transform.location.distance(ego_location) > max_distance:
                continue
            target_wpt = self._map.get_waypoint(target_transform.location, lane_type=carla.LaneType.Any)

            if target_wpt.lane_id == -1 and target_wpt.transform.location.y > 9:
                target_lane_id = -4
            else:
                target_lane_id = target_wpt.lane_id

            if target_lane_id == self._vehicle_wp.lane_id - 1:
                target_vehicle_list.append(target_vehicle)
                target_vehicle_x.append(target_transform.location.x)
        
        # range vehciles on the target lane according to their locations
        ranged_target_vehicle_list = []
        for _ in target_vehicle_list:
            dis = max(target_vehicle_x)
            index = target_vehicle_x.index(dis)
            ranged_target_vehicle_list.append(target_vehicle_list[index])
            target_vehicle_x[index] = -5000

        # get vehicle gaps
        gap_list = []
        gap_loc_dict = {}
        score_dict = {}
        ranged_target_vehicle_list.insert(0,None)
        ranged_target_vehicle_list.append(None)
        for i in range(len(ranged_target_vehicle_list)-1):
            gap_list.append((ranged_target_vehicle_list[i],ranged_target_vehicle_list[i+1]))
        
        # get speed range in control
        _, preceding_vehicle, _ = self._vehicle_obstacle_detected(vehicle_list, max_distance, up_angle_th=90)
        time_gap = self.get_time_gap(preceding_vehicle)
        _, _, next_vel = self._local_planner.run_step(preceding_vehicle, time_gap)
        upper_bound = np.linalg.norm([next_vel.x,next_vel.y,next_vel.z])
        dec_vel = max(min([self._behavior.max_speed, self._speed_limit - self._behavior.speed_lim_dist]) - 20, 0) / 3.6
        middle = max(min([self._behavior.max_speed, self._speed_limit - self._behavior.speed_lim_dist]) - 10, 0) / 3.6
        lower_bound = min(upper_bound, dec_vel)
        vel_list = [upper_bound]
        vehicle_vel = get_speed(self._vehicle)/3.6
        if vehicle_vel > lower_bound and vehicle_vel < upper_bound:
            vel_list.append(vehicle_vel)
        if middle < upper_bound and middle < vehicle_vel:
            vel_list.append(middle)
        if lower_bound < upper_bound:
            vel_list.append(lower_bound)
        
        # base parameters
        lambda_factor = self._lambda_factor
        t_lane_change = 1
        acc_bound = self._safe_acc
        t_bound = max((self._off_ramp_influence_range_x[1] - ego_location.x) / vehicle_vel, 1)
        miu = 0.8
        max_vehicle_vel = min([self._behavior.max_speed, self._speed_limit - self._behavior.speed_lim_dist]) / 3.6
        lon_controller = self._local_planner._vehicle_controller._lon_controller
        
        # score each gap in different target velocity
        for current_vel in vel_list:
            for gap in range(len(gap_list)):
                front_vehicle, back_vehicle = gap_list[gap]
                if front_vehicle:
                    front_loc = front_vehicle.get_location()
                    front_vel = get_speed(front_vehicle) / 3.6
                    time_gap = self.get_time_gap(front_vehicle)
                    front_type = self._vehicle_register.id_type[front_vehicle.id] if front_vehicle.id in self._vehicle_register.id_type else 'hdv'
                else:
                    front_loc = ego_location + carla.Location(x=max_distance/2,y=3.75)
                    front_vel = vehicle_vel
                    time_gap = self._behavior.time_gap_acc
                    front_type = 'hdv'
                if back_vehicle:
                    back_loc = back_vehicle.get_location()
                    back_vel = get_speed(back_vehicle) / 3.6
                else:
                    back_loc = ego_location + carla.Location(x=-max_distance/2,y=3.75)
                    back_vel = vehicle_vel
                
                distance = compute_distance(front_loc, back_loc) - 5
                half_distance = distance/2-2.5
                gap_loc_x = (front_loc.x, back_loc.x)
                if gap_loc_x not in gap_loc_dict:
                    gap_loc_dict[gap] = gap_loc_x
                lon_controller.time_gap = time_gap
                if front_type == 'cav':
                    acceleration = lon_controller._cacc(front_vel, current_vel, half_distance)
                else:
                    acceleration = lon_controller._acc(front_vel, current_vel, half_distance)
                safe_distance = lambda_factor*(back_vel-current_vel)*t_lane_change + (1-lambda_factor)*max_vehicle_vel*t_lane_change
                t = ((front_loc.x + back_loc.x)/2 - ego_location.x) / (current_vel - (front_vel + back_vel) / 2 + 0.0001)
                
                if acceleration > acc_bound and half_distance >= safe_distance and t <= t_bound and t > 0:
                    score = miu * (np.clip(acceleration,acc_bound,0)/acc_bound) + (1-miu) * (t/t_bound)
                    if score not in score_dict:
                        score_dict[score] = (gap, current_vel)
        
        scores = list(score_dict.keys())
        if scores:
            min_score = min(scores)
            target_gap, target_vel = score_dict[min_score]
            target_loc_x = gap_loc_dict[target_gap]
            if target_vel == upper_bound:
                target_speed = min([self._behavior.max_speed, self._speed_limit - self._behavior.speed_lim_dist])
            else:
                target_speed = target_vel * 3.6
            # target_speed = max(min([self._behavior.max_speed, self._speed_limit - self._behavior.speed_lim_dist]) - 20, 0)
            self._target_speed = target_speed
            self._local_planner.set_speed(target_speed)

        lane_change_flag = self.check_lane_change_condition_in_off_ramp(vehicle_list,max_distance,lane_change_wp)
        return lane_change_flag

        #     if lane_change_flag:
        #         return target_loc_x[1] <= ego_location.x <= target_loc_x[0]
        #     else:
        #         return False
        # else:
        #     return False
    
    def run_step(self, current_time=0, debug=False):
        """
        Execute one step of navigation.

            :param current_time: current time in the simulation
            :param debug: boolean for debugging
            :return control: carla.VehicleControl
        """
        self._update_information(current_time)

        # 1: Red lights and stops behavior
        if self.traffic_light_manager():
            self._last_lane_id = self._vehicle_wp.lane_id
            return self.emergency_stop()

        # 2: Pedestrian avoidance behaviors
        walker_state, walker, w_distance = self.pedestrian_avoid_manager(self._vehicle_wp)

        if walker_state:
            # Distance is computed from the center of the two cars,
            # we use bounding boxes to calculate the actual distance
            distance = w_distance - max(
                walker.bounding_box.extent.y, walker.bounding_box.extent.x) - max(
                    self._vehicle.bounding_box.extent.y, self._vehicle.bounding_box.extent.x)

            # Emergency brake if the warker is very close.
            if distance < self._behavior.braking_distance:
                self._last_lane_id = self._vehicle_wp.lane_id
                return self.emergency_stop()

        # 3: Interact with other vehicles
        
        # detect vehicle interrupting the movement
        max_distance = max(self._behavior.min_proximity_threshold, self._speed_limit + self._base_vehicle_threshold) # vehicle at least has 3.6s to react
        vehicle_list = self._world.get_actors().filter("*vehicle*")
        def dist(v): return v.get_location().distance(self._vehicle_wp.transform.location)
        vehicle_list = [v for v in vehicle_list if dist(v) < max_distance and v.id != self._vehicle.id]

        # get preceding and rear vehicle in the same lane
        _, pre_vehicle, _ = self._vehicle_obstacle_detected(vehicle_list, max_distance, up_angle_th=90)
        _, rear_vehicle, _ = self._vehicle_obstacle_detected(vehicle_list, max_distance, low_angle_th=90, up_angle_th=180)
        self._preceding_vehicle = pre_vehicle
        self._rear_vehicle = rear_vehicle

        vehicle_state, vehicle, distance = self.collision_and_car_avoid_manager(max_distance,vehicle_list)

        if vehicle_state:
            
            # use bounding boxes to calculate the actual distance
            distance = distance - max(vehicle.bounding_box.extent.y, vehicle.bounding_box.extent.x) - max(
                    self._vehicle.bounding_box.extent.y, self._vehicle.bounding_box.extent.x)

            # Emergency brake if the vehicle is very close.
            if distance < self._behavior.braking_distance:
                if self._emergency_stop_period >= self._emergency_stop_period_threshold and self._available_destination and \
                    self._direction in [RoadOption.CHANGELANELEFT, RoadOption.CHANGELANERIGHT]:
                    self.change_to_available_destination()
                else:
                    self._emergency_stop_period += self._int_vehicle.dt
                    self._last_lane_id = self._vehicle_wp.lane_id
                    return self.emergency_stop()
            else:
                self._emergency_stop_period = 0
        
        # vehicle plans to follow the lane
        if self._direction != RoadOption.CHANGELANELEFT and self._direction != RoadOption.CHANGELANERIGHT:

            # vehicle plans to go off ramp and in the influence section
            if self.lane_change_in_off_ramp_judgement():
                control, transform, velocity = self.lane_change_in_off_ramp(max_distance, vehicle_list, debug)
            
            # perform car-following and discretionary lane change
            else:
                control, transform, velocity = self.discretionary_lane_change(max_distance, vehicle, distance, vehicle_list, debug)

        # vehicle plans to change lane
        else:
            if self._deactivate_mandatory_lane_change and self._available_destination:
                _, preceding_vehicle, _ = self._vehicle_obstacle_detected(vehicle_list, max_distance, up_angle_th=90)
                control, transform, velocity = self.go_to_next_destination(preceding_vehicle)
                self._target_vehicle = preceding_vehicle
            else:
                control, transform, velocity = self.mandatory_lane_change(max_distance, vehicle_list, debug)

        # save last vehicle waypoint
        self._last_lane_id = self._vehicle_wp.lane_id

        return (control, transform, velocity)
    
    def check_other_lane_change_manevers(self,lane_change_flag,lane_change_wp,vehicle_list):
        """
        This function check vehicles not on the current lane and target lane has the lane change intension
        to the target lane based on its agent recorded by the vehicle register when the ego vehicle plans to 
        change lane
            :param lane_change_flag: whether the ego vehicle plans to change lane
            :param lane_change_wp: carla.Waypoint on the target lane if it plans to change lane
            :param vehicle_list: list of carla.Vehicle to be detected
            :return lane_change_flag: new lane_change_flag
        """
        current_vel = get_speed(self._vehicle)/3.6
        safe_dis = current_vel * 1.5 + 7
        if not lane_change_flag or not lane_change_wp:
            return lane_change_flag
        target_lane_id = lane_change_wp.lane_id
        for vehicle in vehicle_list:
            vehicle_id = vehicle.id
            if vehicle_id in self._vehicle_register.id_agent:
                agent = self._vehicle_register.id_agent[vehicle_id]
                lane_id = agent._lane_id
                if lane_id != self._lane_id and lane_id != target_lane_id and agent._target_lane_id == target_lane_id:
                    if compute_distance(agent._vehicle_wp.transform.location,self._vehicle_wp.transform.location) <= safe_dis:
                        return False
        return lane_change_flag
    
    def change_to_available_destination(self):
        """
        moving ahead by changing destination to avoid blocking the way
            :return bool: whether the manuver is conducted
        """
        location = self._vehicle.get_location()
        if self._off_ramp:
            return False
        elif self._lane_id == -1 and not (location.x>0 and location.y>9):
            destination = self._available_destination[0]
        elif self._lane_id == -2:
            destination = self._available_destination[1]
        elif self._lane_id == -3:
            destination = self._available_destination[2]
        else:
            return False
        self.set_destination(destination.location,location,True)
        return True

    def emergency_stop(self):
        """
        Overwrites the throttle a brake values of a control to perform an emergency stop.
        The steering is kept the same to avoid going out of the lane when stopping during turns

            :param speed (carl.VehicleControl): control to be modified
        """
        control = carla.VehicleControl()
        control.throttle = 0
        control.brake = self._max_brake
        control.hand_brake = False
        
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
        
        return (control, transform, velocity)
    
    def is_within_platoon(self, target_vehicle=None):
        """
        This module checks whether a vehicle is within a platoon
            :param target_vehicle: vehicle to be checked
            :return platoon: platoon that the vehicle is in, None if the vehicle is not within a platoon
        """
        if not target_vehicle:
            target_vehicle = self._vehicle
        if target_vehicle.id in self._vehicle_register.id_platoon:
            return self._vehicle_register.id_platoon[target_vehicle.id]
        else:
            return None
    
    def check_join_platoon(self,target_vehicle):
        """
        This module checks whether a vehicle can join a platoon
            :param target_vehicle: vehicle plans to join
            :return target_platoon: platoon that the vehicle can join, None otherwise
        """
        if target_vehicle:
            target_platoon = self.is_within_platoon(target_vehicle)
            if target_platoon:
                success = target_platoon.vehicle_merge(self._vehicle)
                if success:
                    return target_platoon
        return None
    
    def form_platoon_judgement(self,target_vehicle):
        """
        Check if can form a platoon with two adjacent vehicles
            :param target_vehicle: vehicle in front of the ego vehicle in the same lane
            :return bool: whether can it forms a platoon
        """
        if not target_vehicle:
            return False
        if self.is_within_platoon(target_vehicle):
            return False
        if target_vehicle.id in self._vehicle_register.id_agent:
            target_off_ramp = self._vehicle_register.id_agent[target_vehicle.id]._off_ramp
        else:
            target_off_ramp = True
        if target_off_ramp or self._off_ramp:
            return False
        location = self._vehicle.get_location()
        y_bound = 9
        if self._lane_id == -1 and location.y < y_bound:
            return True
        else:
            return False
    
    def form_platoon(self,target_vehicle):
        """
        Form a platoon with two adjacent vehicles
            :param target_vehicle: vehicle in front of the ego vehicle in the same lane
            :return platoon: platoon formed by the vehicles
        """
        if self.form_platoon_judgement(target_vehicle):
            dis = compute_distance(target_vehicle.get_location(), self._vehicle.get_location())
            equ_dis = 7 + 0.6 * get_speed(self._vehicle)/3.6
            if dis < 2*equ_dis:
                platoon = Platoon(target_vehicle,self._vehicle,self._vehicle_register)
                print('form platoon')
                return platoon
        return None
    
    def platoon_control(self,control,transform,velocity,target_vehicle):
        """
        This module controls platoon behaviors like platoon merging, vehicle merges into a platoon,
        a member diverges from a platoon, platoon longitudinal control and so on.
            :param control: initial control command performed by agent of the ego vehicle without considering the platoon
            :param target_vehicle: preceding vehicle, None if it does not exist
            :return control: carla.VehicleControl considering platoon control behaviors
        """
        if self._activate_platoon_control:
            init_platoon = self.is_within_platoon()
            if self._lane_change_flag:
                # lane change and diverge from the platoon
                if init_platoon:
                    init_platoon.vehicle_diverge(self._vehicle)
            else:
                # in the platoon
                if init_platoon:
                    
                    # member: follow platoon control and check preceding vehicle is in the platoon
                    if self._vehicle.id != init_platoon.leader.id:
                        if target_vehicle and not self.is_within_platoon(target_vehicle):
                            v_control, _, v_velocity = init_platoon.run_step(self._vehicle,control,transform,velocity)
                            v_throttle = v_control.throttle
                            v_velocity_norm = (v_velocity.x**2+v_velocity.y**2+v_velocity.z**2)**0.5
                            velocity_norm = (velocity.x**2+velocity.y**2+velocity.z**2)**0.5
                            if v_throttle - control.throttle > 0.3 or control.brake > 0:
                                init_platoon.vehicle_diverge(self._vehicle)
                            elif v_velocity_norm > velocity_norm + 0.5:
                                init_platoon.vehicle_diverge(self._vehicle)
                        else:
                            control, transform, velocity = init_platoon.run_step(self._vehicle,control,transform,velocity)
                    
                    # leader: manage platoon merge and vehicle merge
                    else:
                        if target_vehicle:
                            target_platoon = self.is_within_platoon(target_vehicle)
                            if target_platoon:
                                success = init_platoon.platoon_merge(target_platoon)
                                if success:
                                    control, transform, velocity = target_platoon.run_step(self._vehicle,control,transform,velocity)
                            else:
                                success = init_platoon.vehicle_merge(target_vehicle)
                                if success:
                                    control, transform, velocity = init_platoon.run_step(self._vehicle,control,transform,velocity)
                
                # not in the platoon, check if it can join a platoon
                else:
                    target_platoon = self.check_join_platoon(target_vehicle)
                    if target_platoon:
                        control, transform, velocity = target_platoon.run_step(self._vehicle,control,transform,velocity)
                    else:
                        present_platoon = self.form_platoon(target_vehicle)
                        if present_platoon:
                            control, transform, velocity = present_platoon.run_step(self._vehicle,control,transform,velocity)
        return (control, transform, velocity)
    
    def cal_vehicle_forward_vel(self):
        """
        Calculate velocity of the ego vehicle on the current lane direction
            :param vehicle_forward_vel: velociy on the current lane direction in m/s
        """
        lane_transform = self._vehicle_wp.transform
        vector = lane_transform.get_forward_vector()
        lane_vector = np.array([vector.x, vector.y, vector.z])
        norm_lane_vector = np.linalg.norm(lane_vector)
        velocity = self._vehicle.get_velocity()
        vel_vector = np.array([velocity.x, velocity.y, velocity.z])
        norm_vel_vector = np.linalg.norm(vel_vector)
        if norm_vel_vector < 1e-3:
            return 0.
        cos_angle = np.clip(np.dot(lane_vector, vel_vector) / (norm_lane_vector * norm_vel_vector), -1., 1.)
        vehicle_forward_vel = abs(cos_angle) * norm_vel_vector
        return vehicle_forward_vel
    
    def follow_current_lane_when_change_lane(self, control, transform, velocity, preceding_vehicle, time_gap):
        """
        Follow preceding vehicle in the current lane when chenging lane to avoid crash if the preceding
        vehicle also performs a lane change manuver
            :param lane_change_flag: bool for whether change lane
            :param control: control calculated in front steps
            :param transform: next transform of the vehicle
            :param velocity: next velocity of the vehicle
            :param preceding vehicle: preceding vehicle in the current lane
            :param time_gap: desired time gap
            :return control: carla.VehicleControl considering two vehicles on two lane seperately
            :return transform: carla.Transform considering two vehicles on two lane seperately
            :return velocity: carla.Vector3D considering two vehicles on two lane seperately
        """
        if self._lane_change_flag:
            control_pre, transform_pre, velocity_pre = self._local_planner.run_step(preceding_vehicle, time_gap)
            if control_pre.throttle < control.throttle:
                control = control_pre
            velocity_pre_norm = (velocity_pre.x**2+velocity_pre.y**2+velocity_pre.z**2)**0.5
            velocity_norm = (velocity.x**2+velocity.y**2+velocity.z**2)**0.5
            if velocity_pre_norm < velocity_norm:
                transform = transform_pre
                velocity = velocity_pre
        return control, transform, velocity
    
    def check_lane_change_condition_in_the_front(self,preceding_vehicle,front_vehicle):
        """
        Check if there is a chance when the ego vehicle can change lane to get a higher speed
        based on the preceding and front vehicle in DLC mode
            :param preceding_vehicle: preceding vehicle on the current lane
            :param front_vehicle: front vehicle on the target lane
            :return bool: whether the ego vehicle can change lane
        """
        lon_controller = self._local_planner._vehicle_controller._lon_controller
        preceding_time_gap = self.get_time_gap(preceding_vehicle)
        preceding_acc = lon_controller.run_step(self._target_speed, preceding_vehicle, preceding_time_gap)
        front_time_gap = self.get_time_gap(front_vehicle)
        front_acc = lon_controller.run_step(self._target_speed, front_vehicle, front_time_gap)
        if front_acc > preceding_acc:
            return True
        else:
            return False