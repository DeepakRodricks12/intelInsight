import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
from collections import deque

# Define a simple neural network model for DQN
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Simple Environment Simulation
class SimpleEnv:
    def __init__(self):
        self.state = np.random.rand(2)  # 2D observation space
        self.done = False

    def step(self, action):
        # Reward based on distance to (0.5, 0.5)
        reward = -np.linalg.norm(self.state - np.array([0.5, 0.5]))
        self.state = np.random.rand(2)  # New random state
        self.done = np.random.rand() < 0.1  # Randomly end an episode
        return self.state, reward, self.done

    def reset(self):
        self.state = np.random.rand(2)
        self.done = False
        return self.state

# FGSM Attack Function
def fgsm_attack(observations, epsilon):
    # Compute the perturbation
    perturbation = epsilon * torch.sign(observations)
    perturbed_observations = observations + perturbation
    return torch.clamp(perturbed_observations, 0, 1)

# DQN Agent
class DQNAgent:
    def __init__(self, input_size, output_size):
        self.model = DQN(input_size, output_size)
        self.memory = deque(maxlen=10000)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()
        self.gamma = 0.99
        self.epsilon = 1.0  # Initial epsilon for exploration
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.batch_size = 32

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(2)  # Random action
        state_tensor = torch.tensor(state, dtype=torch.float32)
        q_values = self.model(state_tensor)
        return torch.argmax(q_values).item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in batch:
            state_tensor = torch.tensor(state, dtype=torch.float32)
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32)

            target = reward
            if not done:
                target += self.gamma * torch.max(self.model(next_state_tensor)).item()

            target_f = self.model(state_tensor)
            target_f[action] = target

            # Train the model
            self.optimizer.zero_grad()
            loss = self.loss_fn(self.model(state_tensor), target_f)
            loss.backward()
            self.optimizer.step()

        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Training Loop with Visualization
def train_dqn(num_episodes=500, epsilon=0.1):
    env = SimpleEnv()
    agent = DQNAgent(input_size=2, output_size=2)  # 2D observation and action space
    total_rewards = []  # List to store total rewards for each episode

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0

        while True:
            # Get the action from the agent
            action = agent.act(state)

            # Step in the environment
            next_state, reward, done = env.step(action)

            # Store experience in memory
            agent.remember(state, action, reward, next_state, done)

            total_reward += reward
            state = next_state

            # Perform replay
            agent.replay()

            if done:
                break

        total_rewards.append(total_reward)  # Store total reward for this episode
        print(f"Episode {episode + 1}: Total Reward: {total_reward}")

        # Apply FGSM on the last observation
        perturbed_state = fgsm_attack(torch.tensor(state, dtype=torch.float32), epsilon)

        # Evaluate against perturbed state
        with torch.no_grad():
            perturbed_action = agent.act(perturbed_state.numpy())
            print(f"Perturbed Action: {perturbed_action}")

    # Visualization
    plt.plot(total_rewards)
    plt.title('Total Rewards over Episodes')
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.grid()
    plt.show()

# Run the training
if __name__ == "__main__":
    train_dqn()
