"""
This script generates a mixed traffic with a certain number of CAVs and HDVs traveling
in the world with random spawn points and destinations.
"""

import os
import sys
import carla
from numpy import random

try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/carla')
except IndexError:
    pass

from agents.navigation.behavior_agent import BehaviorAgent


def main():
    try:
        synchronous_master = True
        vehicles_list = []
        vehicles_list_hdv = []
        vehicles_list_cav = []
        number_of_vehicles = 6
        MPR_of_cav = 0.5
        random_seed = 1
        random.seed(random_seed)
        
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)

        # Retrieve the world that is currently running
        world = client.get_world()

        origin_settings = world.get_settings()

        # set sync mode
        settings = world.get_settings()
        if synchronous_master:
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05
            world.apply_settings(settings)

        blueprint_library = world.get_blueprint_library()
        blueprints = blueprint_library.filter('vehicle.*.*')
        blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
        blueprints = sorted(blueprints, key=lambda bp: bp.id)

        spawn_points = world.get_map().get_spawn_points()
        number_of_spawn_points = len(spawn_points)

        if number_of_vehicles <= number_of_spawn_points:
            random.shuffle(spawn_points)
        else:
            number_of_vehicles = number_of_spawn_points
        number_of_cav = int(number_of_vehicles * MPR_of_cav)
        number_of_hdv = number_of_vehicles - number_of_cav

        # --------------
        # Spawn vehicles
        # --------------
        hero = False
        for n, transform in enumerate(spawn_points):
            if n >= number_of_vehicles:
                break
            blueprint = random.choice(blueprints)
            if blueprint.has_attribute('color'):
                if n < number_of_hdv:
                    # color = blueprint.get_attribute('color').recommended_values[0]
                    color = '0, 0, 0'
                else:
                    # color = blueprint.get_attribute('color').recommended_values[1]
                    color = '255, 255, 255'
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            if hero:
                blueprint.set_attribute('role_name', 'hero')
                hero = False
            else:
                blueprint.set_attribute('role_name', 'autopilot')
            # spawn the vehicles
            vehicle = world.spawn_actor(blueprint, transform)
            vehicles_list.append(vehicle)
            if n < number_of_hdv:
                vehicles_list_hdv.append(vehicle)
            else:
                vehicles_list_cav.append(vehicle)

        # we need to tick the world once to let the client update the spawn position
        world.tick()
        print('spawned %d HDV and %d CAV, press Ctrl+C to exit.' % (len(vehicles_list_hdv), len(vehicles_list_cav)))

        # --------------
        # Control HDV
        # --------------
        traffic_manager = client.get_trafficmanager(8000)
        traffic_manager.set_global_distance_to_leading_vehicle(2.5)
        traffic_manager.set_random_device_seed(random_seed)
        traffic_manager.global_percentage_speed_difference(0)

        for vehicle in vehicles_list_hdv:
            vehicle.set_autopilot(True, traffic_manager.get_port())               
        
        # --------------
        # Control CAV
        # --------------
        agents_list = []
        for vehicle in vehicles_list_cav:
            # create the behavior agent
            agent = BehaviorAgent(vehicle, behavior='normal')

            # set the destination spot
            random.shuffle(spawn_points)

            # to avoid the destination and start position same
            if spawn_points[0].location != agent._vehicle.get_location():
                destination = spawn_points[0]
            else:
                destination = spawn_points[1]

            # generate the route
            agent.set_destination(destination.location, vehicle.get_location())
            agents_list.append(agent)
        
        while True:
            for n, agent in enumerate(agents_list):
                if len(agent._local_planner._waypoints_queue)<1:
                    random.shuffle(spawn_points)
                    # to avoid the destination and start position same
                    if spawn_points[0].location != agent._vehicle.get_location():
                        destination = spawn_points[0]
                    else:
                        destination = spawn_points[1]
                    # generate the route
                    agent.set_destination(destination.location)

                vehicle = vehicles_list_cav[n]
                speed_limit = vehicle.get_speed_limit()
                agent.get_local_planner().set_speed(speed_limit)

                control = agent.run_step()
                vehicle.apply_control(control)
            
            world.tick()

    finally:
        world.apply_settings(origin_settings)
        print('destroying %d HDV and %d CAV.' % (len(vehicles_list_hdv), len(vehicles_list_cav)))
        client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(' - Exited by user.')