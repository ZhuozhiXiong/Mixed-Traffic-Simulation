""" This module contains PID controllers to perform lateral and longitudinal control. """

from collections import deque
import math
import numpy as np
import carla
from automated_agents.misc import get_speed
from automated_agents.misc import compute_distance


class VehicleController():
    """
    VehicleController is the combination of two controllers
    (lateral and longitudinal) to perform the
    low level control a vehicle from client side
    """

    def __init__(self, int_vehicle, offset=0):
        """
        Constructor method.

        :param int_vehicle: actor to apply to lower controller onto
        :param offset: If different than zero, the vehicle will drive displaced from the center line.
        Positive values imply a right offset while negative ones mean a left one. Numbers high enough
        to cause the vehicle to drive through other lanes might break the controller.
        """

        self.max_brake = 0.85
        self.max_throt = 0.7
        self.max_steer = 0.8

        self._vehicle = int_vehicle.vehicle
        self._vehicle_register = int_vehicle.vehicle_register
        self._dt = int_vehicle.dt
        self.past_steering = self._vehicle.get_control().steer
        self._lon_controller = LongitudinalController(int_vehicle)
        self._lat_controller = LateralController(int_vehicle, offset)

    def run_step(self, target_speed, waypoint, target_vehicle, time_gap):
        """
        Execute one step of control invoking both lateral and longitudinal
        PID controllers to reach a target waypoint or follow a vehicle
        at a given target_speed.

            :param target_speed: desired vehicle speed in km/h
            :param waypoint: target location encoded as a waypoint
            :param target_vehicle: carla.vechile to follow or None
            :param time_gap: desired time gap (s)
            :return control: carla.VehicleControl performed to the vehicle
            :return current_transform: carla.Transform in the next tick
            :return vel_vector3d: desired velocity in carla.Vector3D in the next tick
        """
        
        def calculate_output(x,y):
            """ This function predicts vehicular command based on current velocity and desired acceleration """
            return 0.2613+0.006933*x+0.04656*y+0.00595*x*y+0.002213*y**2

        acceleration = self._lon_controller.run_step(target_speed, target_vehicle, time_gap)
        desired_steering = self._lat_controller.run_step(waypoint)
        current_vel = get_speed(self._vehicle) / 3.6
        control = carla.VehicleControl()

        if current_vel >= 15:
            acceleration = np.clip(acceleration,-5,2)
        else:
            acceleration = np.clip(acceleration,-5,5)
        output = calculate_output(current_vel, acceleration)
        
        # combine
        if output >= 0:
            control.throttle = np.clip(output, 0.0, self.max_throt)
            control.brake = 0.0
        else:
            control.throttle = 0.0
            control.brake = np.clip(-output, 0.0, self.max_brake)

        # Steering regulation: changes cannot happen abruptly, can't steer too much.
        if desired_steering > self.past_steering + 0.1:
            current_steering = self.past_steering + 0.1
        elif desired_steering < self.past_steering - 0.1:
            current_steering = self.past_steering - 0.1
        else:
            current_steering = desired_steering

        if current_steering >= 0:
            steering = min(self.max_steer, current_steering)
        else:
            steering = max(-self.max_steer, current_steering)

        control.steer = steering
        control.hand_brake = False
        control.manual_gear_shift = False
        self.past_steering = steering

        # set desired speed, rotation and location
        acceleration = np.clip(acceleration,-5,2)
        current_transform = self._vehicle.get_transform()
        target_vel = current_vel + acceleration * self._dt
        current_location = self._vehicle.get_location()
        target_location = waypoint.transform.location
        vector = np.array([target_location.x-current_location.x,target_location.y-current_location.y,0.])
        vector_norm = np.linalg.norm(vector)
        if vector_norm > 1e-6:
            unit_vector = vector / vector_norm
            vel_vector = unit_vector * target_vel
            vel_vector3d = carla.Vector3D(x=vel_vector[0],y=vel_vector[1],z=vel_vector[2])
            x_vector = np.array([1.,0.,0.])
            rotation_angle = math.degrees(math.acos(np.clip(np.dot(unit_vector, x_vector), -1., 1.)))
            _cross = np.cross(x_vector, unit_vector)
            if _cross[2] < 0:
                rotation_angle *= -1.0
            rotation = carla.Rotation(pitch=0.,yaw=rotation_angle,roll=0.)
        else:
            rotation_angle = 0
            vel_vector3d = carla.Vector3D(x=0.,y=0.,z=0.)
            rotation = carla.Rotation(pitch=0.,yaw=0.,roll=0.)
        
        current_transform.rotation = rotation

        return (control, current_transform, vel_vector3d)


class LongitudinalController():
    """
    LongitudinalController implements longitudinal control using a PID.
    """

    def __init__(self, int_vehicle):
        """
        Constructor method.

            :param int_vehicle: actor to apply longitudinal controller onto
        """
        self._vehicle = int_vehicle.vehicle
        self._behavior = int_vehicle.behavior
        self._dt = int_vehicle.dt
        self._vehicle_register = int_vehicle.vehicle_register
        self._error_buffer_v = deque(maxlen=10)
        self._error_buffer_s = deque(maxlen=10)
        self.gain_v = [0.8, 0, 0]
        self.gain_s = [0.45, 0, 0.25]
        self.gain_acc_s = 0.23 # 0.23
        self.gain_acc_v = 0.07 # 0.07
        self.gain_cacc_p = 0.45 # 0.45
        self.gain_cacc_d = 0.25 # 0.25
        self.a = 1
        self.b = 2.8
        self.time_gap = 1.5

    def run_step(self, target_speed, target_vehicle, time_gap, debug=False):
        """
        Execute one step of longitudinal control to reach a given target speed or follow a preceding vehicle.

            :param target_speed: target speed in Km/h
            :param target_vehicle: carla.vechile to follow or None
            :param debug: boolean for debugging
            :return: throttle control
        """
        
        self.time_gap = time_gap

        ego_speed = get_speed(self._vehicle) / 3.6 # m/s
        target_speed = target_speed / 3.6

        if debug:
            print(f'Current speed = {round(ego_speed * 3.6)} km/h')
        
        output = self.free_drive(ego_speed,target_speed)
        
        # delta_v = target_speed - ego_speed
        # output = self._pid_control(delta_v, 'v')
        
        if target_vehicle:

            target_vehicle_speed = get_speed(target_vehicle) / 3.6 # m/s
            
            # Calculate the distance
            ego_transform = self._vehicle.get_transform()
            ego_front_transform = ego_transform
            ego_front_transform.location += carla.Location(
                self._vehicle.bounding_box.extent.x * ego_transform.get_forward_vector())
            
            target_transform = target_vehicle.get_transform()
            target_rear_transform = target_transform
            target_rear_transform.location -= carla.Location(
                target_vehicle.bounding_box.extent.x * target_transform.get_forward_vector())
            
            distance = compute_distance(target_rear_transform.location, ego_front_transform.location)

            if self._behavior.vehicle_type == 'cav':
                # output = self._acc(target_vehicle_speed, ego_speed, distance)
                if self._vehicle_register.id_type[target_vehicle.id] == 'cav':
                    output = self._cacc(target_vehicle_speed, ego_speed, distance)
                else:
                    output = self._acc(target_vehicle_speed, ego_speed, distance)

                # standard automatic emergency braking system
                if (ego_speed**2 - target_vehicle_speed**2)/(2*distance) >= 4:
                    output = -4
            else:
                equ_dis = self.time_gap * ego_speed + self._behavior.stand_still_dis
                if distance < 2 * equ_dis:
                    output = self._IDM_for_car_following(target_speed, target_vehicle_speed, ego_speed, distance)
            
            if output >= 0 and ego_speed > target_speed:
                if self._behavior.vehicle_type == 'cav':
                    output = self.free_drive(ego_speed,target_speed)
                else:
                    output = 0
        
        return output
    
    def free_drive(self,ego_speed,target_speed):
        # if self._behavior.vehicle_type == 'cav':
        #     delta_v = target_speed - ego_speed
        #     output = self._pid_control(delta_v, 'v')
        # else:
        #     output = self.a * (1 - (ego_speed/target_speed)**4)
        delta_v = target_speed - ego_speed
        output = self._pid_control(delta_v, 'v')
        return output
    
    def _IDM_for_car_following(self, target_vel, target_vehicle_vel, ego_vel, distance):
        """
        IDM car-following model

            :param target_vel: desired velocity of the vehicle in m/s
            :param target_vehicle_vel: velocity of the preceding vehicle in m/s
            :param ego_vel: ego velocity in m/s
            :param distance: distance between rear of the preceding vehicle and the ego vehicle
            :param equ_dis: equilibrium diatance calculated by self.time_gap * ego_speed + self.stand_still_dis
            :return output: desired acceleration
        """
        s_star = max(self.time_gap * ego_vel + ego_vel*(ego_vel-target_vehicle_vel)/(2*(self.a*self.b)**0.5), 0) + self._behavior.stand_still_dis
        output = self.a * (1 - (ego_vel/target_vel)**4 - (s_star/distance)**2)

        return output
    
    def _acc(self, target_vehicle_vel, ego_vel, distance):
        """
        linear feedback control for ACC mode
            :param target_vel: desired velocity of the vehicle in m/s
            :param target_vehicle_vel: velocity of the preceding vehicle in m/s
            :param ego_vel: ego velocity in m/s
            :param distance: distance between rear of the preceding vehicle and the ego vehicle
            :return output: desired acceleration
        """
        equ_dis = self.time_gap * ego_vel + self._behavior.stand_still_dis
        delta_v = target_vehicle_vel - ego_vel
        delta_s = distance - equ_dis
        output = self.gain_acc_s * delta_s + self.gain_acc_v * delta_v
        return output

    def _cacc(self, target_vehicle_vel, ego_vel, distance):
        """
        PD control for CACC mode
            :param target_vel: desired velocity of the vehicle in m/s
            :param target_vehicle_vel: velocity of the preceding vehicle in m/s
            :param ego_vel: ego velocity in m/s
            :param distance: distance between rear of the preceding vehicle and the ego vehicle
            :param equ_dis: equilibrium diatance calculated by self.time_gap * ego_speed + self.stand_still_dis
            :param acc: ego vehicle acceleration in m/s2
            :return output: desired acceleration
        """
        equ_dis = self.time_gap * ego_vel + self._behavior.stand_still_dis
        delta_v = target_vehicle_vel - ego_vel
        delta_s = distance - equ_dis
        output = (self.gain_cacc_p * delta_s + self.gain_cacc_d * delta_v) / (self._dt + self.time_gap*self.gain_cacc_d)
        return output
    
    def _pid_control_for_car_following(self, target_vehicle_vel, ego_vel, distance, equ_dis):
        """
        PID controller for car-following based on delta v and delta s

            :param target_vehicle_vel: velocity of the preceding vehicle in m/s
            :param ego_vel: ego velocity in m/s
            :param distance: distance between rear of the preceding vehicle and the ego vehicle
            :param equ_dis: equilibrium diatance calculated by self.time_gap * ego_speed + self.stand_still_dis
            :return output: desired acceleration
        """
        delta_v = target_vehicle_vel - ego_vel
        delta_s = distance - equ_dis
        output_v = self._pid_control(delta_v, 'v')
        output_s = self._pid_control(delta_s, 's')
        output = output_v + output_s
        return output

    def _pid_control(self, error, flag):
        """
        Estimate the throttle/brake of the vehicle based on the PID equations

            :param target_speed:  target speed in Km/h
            :param current_speed: current speed of the vehicle in Km/h
            :return output: desired acceleration
        """

        if flag == 'v':
            gain = self.gain_v
            error_buffer = self._error_buffer_v
        else:
            gain = self.gain_s
            error_buffer = self._error_buffer_s
        
        error_buffer.append(error)

        if len(error_buffer) >= 2:
            _de = (error_buffer[-1] - error_buffer[-2]) / self._dt
            _ie = sum(error_buffer) * self._dt
        else:
            _de = 0.0
            _ie = 0.0

        return (gain[0] * error) + (gain[2] * _de) + (gain[1] * _ie)


class LateralController():
    """
    LateralController implements lateral control using a PID.
    """

    def __init__(self, int_vehicle, offset=0):
        """
        Constructor method.

            :param int_vehicle: actor to apply lateral controller onto
            :param offset: distance to the center line. If might cause issues if the value
                is large enough to make the vehicle invade other lanes.
        """
        self._vehicle = int_vehicle.vehicle
        self._k_p = 1 # 1.95
        self._k_i = 0.05
        self._k_d = 0.2
        self._dt = int_vehicle.dt
        self._offset = offset
        self._e_buffer = deque(maxlen=10)

    def run_step(self, waypoint):
        """
        Execute one step of lateral control to steer
        the vehicle towards a certain waypoin.

            :param waypoint: target waypoint
            :return: steering control in the range [-1, 1] where:
            -1 maximum steering to left
            +1 maximum steering to right
        """
        return self._pid_control(waypoint, self._vehicle.get_transform())

    def _pid_control(self, waypoint, vehicle_transform):
        """
        Estimate the steering angle of the vehicle based on the PID equations

            :param waypoint: target waypoint
            :param vehicle_transform: current transform of the vehicle
            :return: steering control in the range [-1, 1]
        """
        # Get the ego's location and forward vector
        ego_loc = vehicle_transform.location
        v_vec = vehicle_transform.get_forward_vector()
        v_vec = np.array([v_vec.x, v_vec.y, 0.0])

        # Get the vector vehicle-target_wp
        if self._offset != 0:
            # Displace the wp to the side
            w_tran = waypoint.transform
            r_vec = w_tran.get_right_vector()
            w_loc = w_tran.location + carla.Location(x=self._offset*r_vec.x,
                                                         y=self._offset*r_vec.y)
        else:
            w_loc = waypoint.transform.location

        w_vec = np.array([w_loc.x - ego_loc.x,
                          w_loc.y - ego_loc.y,
                          0.0])

        wv_linalg = np.linalg.norm(w_vec) * np.linalg.norm(v_vec)
        if wv_linalg == 0:
            _dot = 1
        else:
            _dot = math.acos(np.clip(np.dot(w_vec, v_vec) / (wv_linalg), -1.0, 1.0))
        _cross = np.cross(v_vec, w_vec)
        if _cross[2] < 0:
            _dot *= -1.0

        self._e_buffer.append(_dot)
        if len(self._e_buffer) >= 2:
            _de = (self._e_buffer[-1] - self._e_buffer[-2]) / self._dt
            _ie = sum(self._e_buffer) * self._dt
        else:
            _de = 0.0
            _ie = 0.0

        return np.clip((self._k_p * _dot) + (self._k_d * _de) + (self._k_i * _ie), -1.0, 1.0)