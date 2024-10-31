import gymnasium
import highway_env
from matplotlib import pyplot as plt

# Create the highway environment with 20 vehicles
env = gymnasium.make('highway-v0', render_mode='rgb_array')

# Set the number of vehicles
env.unwrapped.vehicles_count = 20

# Reset the environment to the initial state
obs, info = env.reset()

frames = []

# Run the simulation for 60 seconds (assuming 30 FPS, total of 1800 frames)
for _ in range(60 * 30):
    # Initialize the action for the ego vehicle
    action = env.unwrapped.action_type.actions_indexes["IDLE"]  # Default action

    # Retrieve the ego vehicle, usually the first vehicle in the list
    ego_vehicle = env.unwrapped.road.vehicles[0]  # Assuming the first vehicle is the ego vehicle

    # Simple behavior for the ego vehicle
    if ego_vehicle.speed < ego_vehicle.target_speed:
        action = env.unwrapped.action_type.actions_indexes["ACCELERATE"]
    
    # Apply the action for the ego vehicle
    obs, reward, done, truncated, info = env.step(action)

    # Render and capture the frame
    frame = env.render()
    frames.append(frame)

    # Check if the episode is done or truncated
    if done or truncated:
        break

# Optionally, display the captured frames
for frame in frames:
    plt.imshow(frame)
    plt.axis('off')  # Turn off axis
    plt.show()

# Close the environment
env.close()


