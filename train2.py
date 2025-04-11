import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from app import Game, in_bounds, get_neighbors, manhattan  # Ensure these are defined in your backend module

# Environment wrapper for your snake game in versus mode.
class SnakeEnv:
    def __init__(self):
        self.game = Game()
        self.game.reset(mode="versus")
    
    def reset(self):
        # Reset the game and set up initial state variables.
        self.game.reset(mode="versus")
        self.prev_apple_distance = manhattan(self.game.player_snake[0], self.game.apple)
        self.prev_length = len(self.game.player_snake)
        return self.get_state()
    
    def get_state(self):
        # Create a grid representation with shape (1, grid_width, grid_height) for CNN.
        grid = np.zeros((self.game.grid_width, self.game.grid_height), dtype=np.float32)
        
        # Mark the player snake with 1.
        for cell in self.game.player_snake:
            grid[cell[0], cell[1]] = 1.0
        
        # Mark the AI (A*) snake positions with 2.
        for cell in self.game.ai_snake:
            grid[cell[0], cell[1]] = 2.0
        
        # Mark the apple with 3.
        grid[self.game.apple[0], self.game.apple[1]] = 3.0
        
        # Add a channel dimension (for the CNN), resulting in shape (1, grid_width, grid_height).
        return np.expand_dims(grid, axis=0)
    def flood_fill_area(self, start, obstacles):
        visited = set()
        queue = [start]
        count = 0
        while queue:
            pos = queue.pop(0)
            if pos in visited:
                continue
            visited.add(pos)
            # Stop expanding if this cell is blocked.
            if pos in obstacles:
                continue
            count += 1
            for neighbor in get_neighbors(pos):
                if neighbor not in visited:
                    queue.append(neighbor)
        return count

    def step(self, action):
        # Map the integer action to a direction tuple.
        mapping = {
            0: (0, -1),  # Up
            1: (0, 1),   # Down
            2: (-1, 0),  # Left
            3: (1, 0)    # Right
        }
        self.game.change_player_direction(mapping[action])
    
        # Execute the move for both snakes.
        self.game.move()
    
        # Obtain the next state.
        state = self.get_state()
    
        # Initialize reward and terminal flag.
        reward = 0
        done = False
    
        # ---- Multi-Radius Analysis with Dynamic Scaling & Smooth Reward Shaping ----
        # Define obstacles as all occupied cells by either snake.
        obstacles = set(self.game.player_snake + self.game.ai_snake)
    
        # Use the A* snake's head as the starting point.
        astar_head = self.game.ai_snake[0]
    
        # Flood-fill to count the number of reachable cells from the A* head.
        open_area = self.flood_fill_area(astar_head, obstacles)
        # Compute the maximum reachable area (if no obstacles exist).
        max_area = self.flood_fill_area(astar_head, obstacles=set())
    
        # Avoid division by zero.
        if max_area == 0:
            open_ratio = 1.0
        else:
            open_ratio = open_area / max_area
    
        # Using dynamic scaling and a smoothing factor.
        smooth_scale = 3.0  # Tunable: increases sensitivity in the middle range.
        smooth_factor = np.tanh((1 - open_ratio) * smooth_scale)
        floodfill_scale = 0.5  # Tunable base factor for this bonus.
        floodfill_bonus = floodfill_scale * smooth_factor * len(self.game.player_snake)
    
        # ---- Directional Bonus using the A* snake's turning options ----
        ai_direction = self.game.ai_direction  # (dx, dy)
        left_turn  = (-ai_direction[1], ai_direction[0])
        right_turn = (ai_direction[1], -ai_direction[0])
    
        blocked_turns = 0
        for d in [left_turn, right_turn]:
            next_pos = (astar_head[0] + d[0], astar_head[1] + d[1])
            if (not in_bounds(next_pos)) or (next_pos in obstacles):
                blocked_turns += 1
        direction_bonus_scale = 0.3  # Tunable: reward per blocked turning option.
        directional_bonus = direction_bonus_scale * blocked_turns
    
        # ---- Base and Terminal Rewards ----
        if self.game.game_over:
            done = True
            # Terminal reward adjustment: using lower magnitude values may help with gradient flow.
            reward = 100 if self.game.winner == "Player" else -100
        else:
            reward = 1  # Base reward for surviving a step.
            # Growth bonus: incentivize longer length (consider if this should be scaled down).
            if len(self.game.player_snake) > 1:
                reward += len(self.game.player_snake)
            # Option: Add extra bonus when an apple is eaten (if your game logic lets you detect that).
            # For example, if self.game.apple_eaten: reward += 10
            # Add our bonus components.
                        # ---- Apple Distance Reward ----
            # Compute current distance from player's head to the apple.
            current_distance = manhattan(self.game.player_snake[0], self.game.apple)
            # Reward for reducing the distance. (Positive delta means moving closer.)
            delta_distance = self.prev_apple_distance - current_distance
            # Compute advantage (player length minus opponent length).
            advantage = len(self.game.player_snake) - len(self.game.ai_snake)
            # When behind or neutral, encourage apple pursuit more.
            apple_weight = 2.0 if advantage <= 0 else 1.0
            apple_distance_reward = apple_weight * delta_distance
            reward += apple_distance_reward
    
            # ---- Attack (Boxing) Bonus Conditioned on Advantage ----
            # Increase attack bonus if the player is ahead.
            attack_factor = 1.5 if advantage > 0 else 0.5
            attack_bonus = attack_factor * (floodfill_bonus + directional_bonus)
            reward += attack_bonus
    
            # Extra bonus if an apple was just eaten.
            # (Assuming the snake grows when apple is eaten.)
            if len(self.game.player_snake) > self.prev_length:
                reward += 20  # Tunable bonus for apple consumption.
    
            # Update stored values for next step.
            self.prev_apple_distance = current_distance
            self.prev_length = len(self.game.player_snake)

    
        return state, reward, done


# Define a CNN-based DQN network.
class CNN_DQN(nn.Module):
    def __init__(self, grid_width, grid_height, action_size):
        super(CNN_DQN, self).__init__()
        # Convolutional layers to extract spatial features.
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # The feature map keeps the same dimensions due to padding.
        self.fc1 = nn.Linear(64 * grid_width * grid_height, 512)
        self.fc2 = nn.Linear(512, action_size)
    
    def forward(self, x):
        # x is expected to have shape [batch, 1, grid_width, grid_height].
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


# DQNAgent using the CNN_DQN network.
class DQNAgent:
    def __init__(self, grid_width, grid_height, action_size):
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95  # Discount factor.
        self.epsilon = 1.0  # Initial exploration rate.
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995  # Exploration decay per step.
        self.learning_rate = 0.001
        self.batch_size = 64
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Initialize the CNN DQN model.
        self.model = CNN_DQN(grid_width, grid_height, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        # Use epsilon-greedy policy for action selection.
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        # The state from get_state() has shape (1, grid_width, grid_height). Add a batch dimension.
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            act_values = self.model(state_tensor)
        return torch.argmax(act_values, dim=1).item()
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        minibatch = random.sample(self.memory, self.batch_size)
        # Stack states and next_states to form batches.
        states = torch.FloatTensor(np.array([exp[0] for exp in minibatch])).to(self.device)
        actions = torch.LongTensor([exp[1] for exp in minibatch]).to(self.device)
        rewards = torch.FloatTensor([exp[2] for exp in minibatch]).to(self.device)
        next_states = torch.FloatTensor(np.array([exp[3] for exp in minibatch])).to(self.device)
        dones = torch.FloatTensor([float(exp[4]) for exp in minibatch]).to(self.device)
        
        # Compute current Q-values.
        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        # Compute next state maximum Q-values.
        next_q_values = self.model(next_states).max(1)[0]
        target = rewards + self.gamma * next_q_values * (1 - dones)
        
        loss = self.criterion(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Gradually decay epsilon.
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# Training loop for the CNN-based DQN agent.
def train_dqn(episodes):
    def get_death_info(env):
        if env.game.player_died_by:
            return env.game.player_died_by
        return "Unknown"
    
    env = SnakeEnv()
    grid_width = env.game.grid_width
    grid_height = env.game.grid_height
    action_size = 4  # Up, Down, Left, Right.
    
    agent = DQNAgent(grid_width, grid_height, action_size)
    best_snake_length = 0
    player_win_count = 0
    death_stats = {"AI": 0, "Player": 0, "Wall": 0, "Unknown": 0}
    recent_rewards = []
    
    for e in range(episodes):
        state = env.reset()  # state shape: (1, grid_width, grid_height)
        done = False
        total_reward = 0
        step_count = 0
        
        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            step_count += 1
            
            agent.replay()
            
            if done:
                best_snake_length = max(len(env.game.player_snake), best_snake_length)
                if env.game.winner == "Player":
                    player_win_count += 1
                recent_rewards.append(total_reward)
                if len(recent_rewards) > 200:
                    recent_rewards.pop(0)
                avg_recent_reward = np.mean(recent_rewards)
                death_cause = get_death_info(env)
                death_stats[death_cause] += 1
                if (e + 1) % 1 == 0:
                    print(f"Episode {e+1}/{episodes} - Steps: {step_count} - Total Reward: {total_reward:.2f} - "
                          f"Epsilon: {agent.epsilon:.2f} - Best Length: {best_snake_length} - "
                          f"Player Wins: {player_win_count} - Death Causes: {death_stats} - "
                          f"Avg Reward (last 200): {avg_recent_reward:.2f}")
                break

if __name__ == '__main__':
    episodes = 300  # Adjust number of episodes as needed.
    train_dqn(episodes)
