import gymnasium
import highway_env
import numpy as np

# Configure the environment for 5 cars
config = {
    "vehicles_count": 5,  # Number of cars
    "duration": 40,       # Duration of the simulation in seconds (this is internal)
    "policy_frequency": 15,  # Control frequency (15 steps per second)
}

# Create the racetrack environment with the custom configuration
env = gymnasium.make('racetrack-v0', render_mode='human', config=config)

# Reset the environment
obs, info = env.reset()  # Reset and get initial observation
print("Initial Observation Structure:", obs)  # Print the observation structure for diagnosis

# Initialize the number of vehicles variable
num_vehicles = 0

# Check if vehicles are present in the observation
if isinstance(obs, dict) and "vehicles" in obs:
    num_vehicles = len(obs["vehicles"])  # Count the number of vehicles
    print(f"Number of vehicles: {num_vehicles}")

# Set the number of steps for the simulation
num_steps = 50  # Run for 50 steps
steps = 0

# Ensure that there are vehicles before entering the simulation loop
if num_vehicles > 0:
    while steps < num_steps:
        # Get the car's current position and speed
        vehicle = obs["vehicles"][0]  # Get the first vehicle's state for control

        speed = vehicle["speed"]
        heading = vehicle["heading"]  # Current heading of the vehicle in radians
        lane_position = vehicle["lane_position"]  # Position on the lane

        # Simple control logic
        steering = 0.0
        acceleration = 0.0
        
        # Steering control: if the vehicle is off the track, adjust steering
        if lane_position < -0.5:  # Off track to the left
            steering = 0.2  # Turn right
        elif lane_position > 0.5:  # Off track to the right
            steering = -0.2  # Turn left
        else:
            steering = 0.0  # Straighten out if on track

        # Acceleration control: accelerate or brake based on speed
        if speed < 5.0:  # If going slow, accelerate
            acceleration = 1.0
        elif speed > 15.0:  # If going fast, decelerate
            acceleration = -1.0
        else:
            acceleration = 0.5  # Maintain speed

        # Create the action vector
        action = [steering, acceleration]  # [steering, acceleration]
        
        # Step through the environment
        obs, reward, done, truncated, info = env.step(action)
        env.render()  # Render the current state for display

        steps += 1
        if done or truncated:
            obs, info = env.reset()  # Reset if the episode ends early

# Print out the total number of steps taken in the simulation
print(f"Simulation ran for {steps} steps.")

# Keep the window open until the user closes it
#input("Press Enter to close the simulation window...")

# Close the environment
env.close()
