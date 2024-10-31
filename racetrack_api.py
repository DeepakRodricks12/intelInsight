from __future__ import division, print_function, absolute_import

import numpy as np
import numpy.random as random
import gym
from gym import spaces
import matplotlib.pyplot as plt

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import LineType, StraightLane, CircularLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.behavior import IDMVehicle

NEXT_ROAD = {
    ("a", "b"): ("b", "c"),
    ("b", "c"): ("c", "d"),
    ("c", "d"): ("d", "e"),
    ("d", "e"): ("e", "f"),
    ("e", "f"): ("f", "g"),
    ("f", "g"): ("g", "h"),
    ("g", "h"): ("h", "i"),
    ("h", "i"): ("i", "a"),
    ("i", "a"): ("a", "b"),
}

class RaceTrackEnv(AbstractEnv):
    """A lane keeping control task with interaction, in a racetrack-like loop."""

    def __init__(self, params, config: dict = None) -> None:
        # Configure Environment with Opt Parameters
        config = {
            "observation": {
                "type": "OccupancyGrid",
                "features": ['presence', 'on_road'],
                "grid_size": [[-18, 18], [-18, 18]],
                "grid_step": [2, 2],
                "as_image": False,
                "align_to_vehicle_axes": True,
            } if params['obs_dim'][0] == 2 else {
                "type": "GrayscaleObservation",
                "observation_shape": tuple(params['obs_dim'][-2:]),
                "stack_size": params['obs_dim'][0],
                "weights": [0.2989, 0.5870, 0.1140],
                "scaling": 1.75,
            },
            "action": {
                "type": "ContinuousAction",
                "longitudinal": False if params['num_actions'] < 2 else True,
                "lateral": True,
                "dynamical": False,
                "steering_range": [-np.pi / 4, np.pi / 4],
            },
            "all_random": params['all_random'],
            "spawn_vehicles": params['spawn_vehicles'],
            "random_lane": params['random_lane'],
            "duration": 200,
            "simulation_frequency": 15,
            "policy_frequency": 5,
        }
        
        # Default Initialization
        super().__init__(config)
        self.lane = None
        self.lanes = []
        self.trajectory = []
        self.interval_trajectory = []
        self.lpv = None
        
        # Variables for Rewards
        self.agent_current = None
        self.agent_target = None
        self.offroad_counter = 0
        self.offroad_threshold = params['offroad_thresh']

        # Define action and observation spaces
        self.action_space = spaces.Box(low=np.array([-1, -np.pi/4]), high=np.array([1, np.pi/4]), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=255, shape=(params['obs_dim'][-2], params['obs_dim'][-1]), dtype=np.uint8)

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "controlled_vehicles": 1,
            "ego_spacing": 2,
            "collision_reward": -5,
            "action_reward": 0.3,
            "offroad_penalty": -1,
            "lane_centering_cost": 4,
            "subgoal_reward_ratio": 1,
            "screen_width": 1000,
            "screen_height": 1000,
            "centering_position": [0.5, 0.5],
        })
        return config

    def _reward(self, action: np.ndarray) -> float:
        longitudinal, lateral = self.vehicle.lane.local_coordinates(self.vehicle.position)
        if self.agent_target in [None, self.vehicle.lane_index[:2]]:
            self.agent_current = self.vehicle.lane_index[:2]
            self.agent_target = NEXT_ROAD[self.agent_current]

        action_reward = - self.config["action_reward"] * np.linalg.norm(action)
        subgoal_reward = self.config["subgoal_reward_ratio"] * (self.vehicle.lane.length - longitudinal) / self.vehicle.lane.length
        lane_centering_reward = 1 / (1 + self.config["lane_centering_cost"] * (lateral) ** 2)

        reward = lane_centering_reward + action_reward + subgoal_reward
        
        if not self.vehicle.on_road or not self._reward_laning():
            reward = self.config["offroad_penalty"]
        if self.vehicle.crashed:
            reward = self.config["collision_reward"]

        if not self.vehicle.on_road:
            self.offroad_counter += 1
        else:
            self.offroad_counter = 0
        
        reward = utils.lmap(reward, [-1, 2], [0, 1])
    
        return reward

    def _is_terminal(self) -> bool:
        return self.vehicle.crashed or self._is_goal() or \
            self.steps >= self.config["duration"] or \
            self.offroad_counter == self.offroad_threshold

    def _reward_laning(self) -> int:
        current_lane = self.road.network.get_closest_lane_index(self.vehicle.position)[:2]
        return current_lane == self.agent_current

    def _is_goal(self) -> bool:
        return self.vehicle.on_road and self.vehicle.lane_index[:2] == ["i", "a"]

    def _reset(self) -> None:
        self.agent_current = None
        self.agent_target = None
        self.offroad_counter = 0
        self._make_road()
        self._make_vehicles()
        return self._get_observation()

    def _make_road(self) -> None:
        net = RoadNetwork()
        speedlimits = [None, 10, 10, 10, 10, 10, 10, 10, 10]
        
        lane = StraightLane([42, 0], [100, 0], line_types=(LineType.CONTINUOUS, LineType.STRIPED), width=5, speed_limit=speedlimits[1])
        net.add_lane("a", "b", lane)

        net.add_lane("a", "b", StraightLane([42, 5], [100, 5], line_types=(LineType.STRIPED, LineType.CONTINUOUS), width=5, speed_limit=speedlimits[1]))

        center1 = [100, -20]
        radii1 = 20
        net.add_lane("b", "c", CircularLane(center1, radii1, np.deg2rad(90), np.deg2rad(-1), width=5, clockwise=False, line_types=(LineType.CONTINUOUS, LineType.NONE), speed_limit=speedlimits[2]))
        net.add_lane("b", "c", CircularLane(center1, radii1 + 5, np.deg2rad(90), np.deg2rad(-1), width=5, clockwise=False, line_types=(LineType.STRIPED, LineType.CONTINUOUS), speed_limit=speedlimits[2]))

        net.add_lane("c", "d", StraightLane([120, -19], [120, -30], line_types=(LineType.CONTINUOUS, LineType.NONE), width=5, speed_limit=speedlimits[3]))
        net.add_lane("c", "d", StraightLane([125, -19], [125, -30], line_types=(LineType.STRIPED, LineType.CONTINUOUS), width=5, speed_limit=speedlimits[3]))

        center2 = [100, -50]
        radii2 = 20
        net.add_lane("d", "e", CircularLane(center2, radii2, np.deg2rad(0), np.deg2rad(-90), width=5, clockwise=True, line_types=(LineType.CONTINUOUS, LineType.NONE), speed_limit=speedlimits[4]))
        net.add_lane("d", "e", CircularLane(center2, radii2 + 5, np.deg2rad(0), np.deg2rad(-90), width=5, clockwise=True, line_types=(LineType.STRIPED, LineType.CONTINUOUS), speed_limit=speedlimits[4]))

        net.add_lane("e", "f", StraightLane([80, -75], [80, -50], line_types=(LineType.CONTINUOUS, LineType.NONE), width=5, speed_limit=speedlimits[5]))
        net.add_lane("e", "f", StraightLane([85, -75], [85, -50], line_types=(LineType.STRIPED, LineType.CONTINUOUS), width=5, speed_limit=speedlimits[5]))

        center3 = [60, -60]
        radii3 = 20
        net.add_lane("f", "g", CircularLane(center3, radii3, np.deg2rad(90), np.deg2rad(-1), width=5, clockwise=False, line_types=(LineType.CONTINUOUS, LineType.NONE), speed_limit=speedlimits[5]))
        net.add_lane("f", "g", CircularLane(center3, radii3 + 5, np.deg2rad(90), np.deg2rad(-1), width=5, clockwise=False, line_types=(LineType.STRIPED, LineType.CONTINUOUS), speed_limit=speedlimits[5]))

        net.add_lane("g", "h", StraightLane([40, -30], [40, 0], line_types=(LineType.CONTINUOUS, LineType.NONE), width=5, speed_limit=speedlimits[6]))
        net.add_lane("g", "h", StraightLane([45, -30], [45, 0], line_types=(LineType.STRIPED, LineType.CONTINUOUS), width=5, speed_limit=speedlimits[6]))

        center4 = [25, 20]
        radii4 = 20
        net.add_lane("h", "i", CircularLane(center4, radii4, np.deg2rad(90), np.deg2rad(-1), width=5, clockwise=False, line_types=(LineType.CONTINUOUS, LineType.NONE), speed_limit=speedlimits[7]))
        net.add_lane("h", "i", CircularLane(center4, radii4 + 5, np.deg2rad(90), np.deg2rad(-1), width=5, clockwise=False, line_types=(LineType.STRIPED, LineType.CONTINUOUS), speed_limit=speedlimits[7]))

        net.add_lane("i", "a", StraightLane([20, 0], [0, 0], line_types=(LineType.CONTINUOUS, LineType.NONE), width=5, speed_limit=speedlimits[8]))
        net.add_lane("i", "a", StraightLane([20, -5], [0, -5], line_types=(LineType.STRIPED, LineType.CONTINUOUS), width=5, speed_limit=speedlimits[8]))

        self.road = Road(network=net, vehicles=[])

    def _make_vehicles(self):
        """Initialize vehicles in the environment."""
        for _ in range(self.config["controlled_vehicles"]):
            lane_id = ("a", "b")  # Specify the lane from which to spawn vehicles
            lane = self.road.network.get_lane(lane_id)  # Get the lane object
            position = random.uniform(0, lane.length)  # Random position along the lane
            
            # Create a vehicle with random attributes
            vehicle = IDMVehicle(
                road=self.road,
                lane_index=lane_id,
                target_speed=10 + random.uniform(-5, 5),  # Random target speed
                position=position  # Random position along the lane
            )
            self.road.vehicles.append(vehicle)
        
        self.vehicle = self.road.vehicles[0]  # Set the first vehicle as the agent

    def render(self, mode='human') -> None:
        """Render the environment."""
        plt.clf()  # Clear the current figure
        plt.imshow(self._get_observation(), cmap='gray')  # Render the observation
        plt.axis('off')  # Turn off the axis
        plt.pause(0.01)  # Pause to allow the plot to update

    def _get_observation(self):
        # Generate a dummy observation for demonstration purposes
        # You can customize this method to return actual observation data
        obs = np.zeros((40, 40), dtype=np.uint8)  # Example observation shape
        # Fill the observation with some data (e.g., vehicle position)
        if hasattr(self, 'vehicle') and self.vehicle is not None:
            x, y = self.vehicle.position.astype(int)
            obs[x % 40, y % 40] = 255  # Example vehicle representation
        return obs

if __name__ == "__main__":
    # Initialize the environment
    env = RaceTrackEnv(params={'obs_dim': (2, 40, 40), 'num_actions': 2, 'all_random': True, 'spawn_vehicles': 1, 'random_lane': False, 'offroad_thresh': 10})
    
    # Reset the environment
    obs = env.reset()
    
    done = False
    while not done:
        action = env.action_space.sample()  # Sample a random action
        obs, reward, done, _ = env.step(action)  # Take a step in the environment
        env.render()  # Render the environment

    env.close()  # Close the environment




