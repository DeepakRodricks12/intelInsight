import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import multiprocessing as mp
import matplotlib.pyplot as plt
import time

# Define the Actor-Critic Model
class A3CModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(A3CModel, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, output_size),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        action_probs = self.actor(x)
        state_value = self.critic(x)
        return action_probs, state_value

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
    perturbation = epsilon * torch.sign(observations)
    perturbed_observations = observations + perturbation
    return torch.clamp(perturbed_observations, 0, 1)

# A3C Worker
class A3CWorker:
    def __init__(self, global_model, input_size, output_size, optimizer, epsilon):
        self.global_model = global_model
        self.local_model = A3CModel(input_size, output_size)
        self.local_model.load_state_dict(self.global_model.state_dict())  # Sync models
        self.optimizer = optimizer
        self.epsilon = epsilon
        self.env = SimpleEnv()

    def train(self, worker_id, num_episodes):
        total_rewards = []

        for episode in range(num_episodes):
            state = self.env.reset()
            done = False
            total_reward = 0

            while not done:
                state_tensor = torch.tensor(state, dtype=torch.float32)
                action_probs, _ = self.local_model(state_tensor)
                action = np.random.choice(len(action_probs), p=action_probs.detach().numpy())
                
                # Apply FGSM on the state
                perturbed_state = fgsm_attack(state_tensor, self.epsilon)

                # Step in the environment with the action
                next_state, reward, done = self.env.step(action)
                total_reward += reward

                # Calculate the advantage and target value
                _, state_value = self.local_model(state_tensor)
                _, next_state_value = self.local_model(torch.tensor(next_state, dtype=torch.float32))

                advantage = reward + (1 - done) * next_state_value.item() - state_value.item()

                # Save the gradients for the global model
                self.optimizer.zero_grad()

                # Calculate actor and critic loss
                actor_loss = -torch.log(action_probs[action]) * advantage
                critic_loss = advantage ** 2
                total_loss = actor_loss + critic_loss

                total_loss.backward()
                for global_param, local_param in zip(self.global_model.parameters(), self.local_model.parameters()):
                    if local_param.grad is not None:
                        global_param._grad = local_param.grad  # Copy gradients

                self.optimizer.step()
                self.local_model.load_state_dict(self.global_model.state_dict())  # Sync after each step
                state = next_state

            total_rewards.append(total_reward)

        return total_rewards

# Separate function for worker process
def worker_process(rewards_queue, worker, worker_id, num_episodes):
    rewards = worker.train(worker_id, num_episodes)
    rewards_queue.put(rewards)

# Main A3C Training Function
def train_a3c(num_workers=4, num_episodes=1000):
    input_size = 2  # State dimension
    output_size = 2  # Action dimension
    global_model = A3CModel(input_size, output_size)
    global_model.share_memory()  # Allow sharing the model's memory for multiprocessing
    optimizer = optim.Adam(global_model.parameters(), lr=0.001)

    workers = []
    rewards_queue = mp.Queue()

    for i in range(num_workers):
        worker = A3CWorker(global_model, input_size, output_size, optimizer, epsilon=0.1)
        p = mp.Process(target=worker_process, args=(rewards_queue, worker, i, num_episodes))
        workers.append(p)

    for p in workers:
        p.start()

    for p in workers:
        p.join()

    # Gather results from the workers
    total_rewards = []
    while not rewards_queue.empty():
        total_rewards.extend(rewards_queue.get())

    # Visualization
    plt.plot(np.convolve(total_rewards, np.ones(50)/50, mode='valid'))  # Moving average
    plt.title('Total Rewards over Episodes')
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.grid()
    plt.show()

# Run the training
if __name__ == "__main__":
    mp.set_start_method('spawn')  # Necessary for some multiprocessing cases
    train_a3c(num_workers=4, num_episodes=500)


