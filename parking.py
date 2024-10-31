import gymnasium 
import highway_env
from matplotlib import pyplot as plt

# Create the highway environment
env = gymnasium.make('highway-v0', render_mode='rgb_array')
env.reset()

# Check available actions in the environment
print("Available actions:", env.unwrapped.action_type.actions_indexes)

# Use IDLE action for the first few steps
idle_action = env.unwrapped.action_type.actions_indexes["IDLE"]

# Execute idle actions in the environment
for _ in range(3):
    obs, reward, done, truncated, info = env.step(idle_action)
    env.render()

# Choose an action to simulate parking
# Here we use IDLE to represent the vehicle being parked
parking_action = idle_action  # or you can choose LANE_LEFT or LANE_RIGHT based on your strategy

# Execute the action that simulates parking
obs, reward, done, truncated, info = env.step(parking_action)
env.render()  # Render the final state after "parking"

# Display the environment's rendered output
plt.imshow(env.render())
plt.show()
