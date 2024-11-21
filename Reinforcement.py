import numpy as np
import matplotlib.pyplot as plt

# Define the environment
class GridWorld:
    def __init__(self, size, start, goal, obstacles):
        self.size = size
        self.start = start
        self.goal = goal
        self.obstacles = obstacles
        self.reset()

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action):
        x, y = self.state
        if action == 0:  # Up
            next_state = (max(x - 1, 0), y)
        elif action == 1:  # Down
            next_state = (min(x + 1, self.size - 1), y)
        elif action == 2:  # Left
            next_state = (x, max(y - 1, 0))
        elif action == 3:  # Right
            next_state = (x, min(y + 1, self.size - 1))

        # If next state is an obstacle, stay in the same place
        if next_state in self.obstacles:
            next_state = self.state

        reward = -1
        done = False

        # Goal reached
        if next_state == self.goal:
            reward = 10
            done = True

        self.state = next_state
        return next_state, reward, done

# Parameters
size = 5
start = (0, 0)
goal = (4, 4)
obstacles = [(2, 2), (3, 2)]

env = GridWorld(size, start, goal, obstacles)

# Initialize Q-table
actions = 4  # Up, Down, Left, Right
q_table = np.zeros((size, size, actions))

# Hyperparameters
episodes = 1000
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 1.0  # Exploration-exploitation tradeoff
epsilon_decay = 0.995
min_epsilon = 0.1

# Training
rewards = []
for episode in range(episodes):
    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        x, y = state
        if np.random.uniform(0, 1) < epsilon:
            action = np.random.choice(actions)  # Explore
        else:
            action = np.argmax(q_table[x, y])  # Exploit

        next_state, reward, done = env.step(action)
        next_x, next_y = next_state

        # Update Q-value
        q_table[x, y, action] = q_table[x, y, action] + alpha * (
            reward + gamma * np.max(q_table[next_x, next_y]) - q_table[x, y, action]
        )

        state = next_state
        total_reward += reward

    rewards.append(total_reward)
    epsilon = max(min_epsilon, epsilon * epsilon_decay)  # Decay epsilon

# Plot rewards
plt.plot(rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Reward Progression')
plt.show()

# Visualize Q-values
print("Learned Q-values:")
for i in range(size):
    for j in range(size):
        print(f"({i}, {j}): {q_table[i, j]}")
