import gymnasium
import highway_env
import numpy as np

# Create the Intersection environment with the specified render mode
env = gymnasium.make("highway-v0", render_mode="human")  # Use "rgb_array" if you prefer that format

# Number of episodes
num_episodes = 10

# Main loop to run the environment
for episode in range(num_episodes):
    done = False
    total_reward = 0
    state, info = env.reset()  # Reset the environment

    while not done:
        # Render the environment
        env.render()

        # Take a random action (you can replace this with your policy)
        action = env.action_space.sample()

        # Step the environment and unpack the return values using a dictionary
        step_result = env.step(action)
        next_state = step_result['observation']
        reward = step_result['reward']
        done = step_result['terminated']  # or step_result['truncated'] if applicable
        info = step_result['info']

        # Accumulate total reward
        total_reward += reward

        # Update state
        state = next_state

    print(f"Episode {episode + 1} finished with total reward: {total_reward}")

# Close the environment
env.close()
