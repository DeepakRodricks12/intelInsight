import gymnasium as gym
import highway_env
import numpy as np

def create_custom_env():
    # Create the highway environment
    env = gym.make("highway-v0")
    return env

def add_incoming_vehicles(env):
    # Create incoming vehicles on the ramp
    for i in range(1, 6):  # Create 5 incoming vehicles
        vehicle = {
            "id": f"incoming_{i}",
            "length": 5,
            "width": 2,
            "max_speed": 20,  # Set a lower speed for incoming vehicles
            "lane_index": 1,  # Set to the access ramp lane
            "position": np.array([-50, i * 2])  # Position them off the highway
        }
        env.vehicles.append(vehicle)

def run_merging_simulation():
    env = create_custom_env()
    env.reset()

    # Adding incoming vehicles directly to the environment
    add_incoming_vehicles(env)

    done = False
    while not done:
        action = env.action_space.sample()  # Random action for testing
        obs, reward, done, info = env.step(action)
        
        env.render()  # Render the environment (if supported)

        # Print the positions and speeds of the ego vehicle
        print(f"Ego Vehicle Position: {env.ego_vehicle.position}, Speed: {env.ego_vehicle.speed}")

if __name__ == "__main__":
    run_merging_simulation()

