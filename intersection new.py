import gymnasium as gym
import highway_env

# Create the intersection environment
env = gym.make("intersection-v0", render_mode="human")

# Configure the underlying environment
env.unwrapped.configure({
    "simulation_frequency": 15,    # Frequency of simulation steps
    "duration": 400,                # Duration of the episode in seconds
    "vehicles_count": 15,          # Number of vehicles in the environment
    "observation": {
        "type": "Kinematics",      # Type of observation space (positions, velocities)
        "vehicles_count": 5,       # Number of vehicles in observation
        "features": ["x", "y", "vx", "vy"],  # Observed features
        "absolute": True           # Absolute coordinates for observations
    },
    "policy_frequency": 2,         # Frequency of policy decisions
    "screen_width": 600,           # Width of the rendering window
    "screen_height": 600,          # Height of the rendering window
    "centering_position": [0.5, 0.5]  # Center of the screen in (x, y) format
})

# Reset the environment
obs, info = env.reset()

# Run the simulation
done = False
while not done:
    # Sample a random action from the action space
    action = env.action_space.sample()
    
    # Step through the environment
    obs, reward, done, truncated, info = env.step(action)
    
    # Render the environment
    env.render()

# Close the environment
env.close()
