import gymnasium as gym
import highway_env
import numpy as np
import matplotlib.pyplot as plt

# Initialize the environment with render mode
env = gym.make('highway-v0', render_mode='human')  # Use 'human' for real-time rendering

# Number of episodes to run
num_episodes = 10
rewards = []

for episode in range(num_episodes):
    total_reward = 0
    state, info = env.reset()  # Reset the environment and get the initial state
    done = False  # Reset done for each episode

    while not done:
        # Choose a random action from the action space
        action = env.action_space.sample()

        # Step the environment with the chosen action
        state, reward, done = env.step(action)  # Change here

        # Accumulate the reward
        total_reward += reward

    rewards.append(total_reward)
    print(f"Episode {episode + 1}/{num_episodes} - Total Reward: {total_reward}")

# Close the environment
env.close()

# Plotting the rewards over episodes
plt.plot(rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Total Reward over Episodes in highway-v0')
plt.show()

