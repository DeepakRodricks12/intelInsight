import gymnasium
import highway_env
import random

# Create a roundabout environment
env = gymnasium.make('roundabout-v0', render_mode='rgb_array')

# Reset the environment to its initial state
obs, info = env.reset()
img = env.render()  # Render the initial state

# Specify the number of steps to run the simulation
num_steps = 100  # Extend the simulation time to 100 steps

# Run a loop for the specified number of steps in the environment
for step in range(num_steps):
    # Randomly select an action from the action space
    action = random.choice(list(env.unwrapped.action_type.actions_indexes.values()))
    
    # Step the environment forward with the selected action
    obs, reward, done, truncated, info = env.step(action)
    
    # Render the environment and retrieve the image
    img = env.render()

    # Check if the rendered image is valid
    if img is None:
        print(f"Warning: Rendered image is None at step {step}. Check the render method.")

# Close the environment
env.close()



