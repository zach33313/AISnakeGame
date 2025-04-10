# training.py

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
        # Reset the game in versus mode and return the initial state.
        self.game.reset(mode="versus")
        self.prev_apple_distance = manhattan(self.game.player_snake[0], self.game.apple)
        self.prev_length = len(self.game.player_snake)
        return self.get_state()
    
    def get_state(self):
        # Create a simple grid representation.
        grid = np.zeros((self.game.grid_width, self.game.grid_height), dtype=np.float32)
        
        # Mark player snake positions with 1.
        for cell in self.game.player_snake:
            grid[cell[0], cell[1]] = 1.0
        
        # Mark AI (A*) snake positions with 2.
        for cell in self.game.ai_snake:
            grid[cell[0], cell[1]] = 2.0
        
        # Mark apple with 3.
        grid[self.game.apple[0], self.game.apple[1]] = 3.0
        
        # Flatten grid to create a state vector.
        return grid.flatten()
    
    # Helper: Flood fill to count reachable open cells from a given position.
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
    '''
    # Updated step() method for the SnakeEnv environment.
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
'''
    def step(self, action):
        # Map action to direction
        mapping = {
            0: (0, -1),  # Up
            1: (0, 1),   # Down
            2: (-1, 0),  # Left
            3: (1, 0)    # Right
        }
        self.game.change_player_direction(mapping[action])
        self.game.move()
        state = self.get_state()
        done = False

        # Get the player's head position and compute distance to the nearest wall.
        head = self.game.player_snake[0]
        # Compute Manhattan distance from head to the closest wall.
        dist_to_wall = min(head[0], self.game.grid_width - 1 - head[0],
                       head[1], self.game.grid_height - 1 - head[1])
        wall_penalty = 0
        # Penalize if too close to a wall (e.g., if distance < 2, apply a penalty).
        if dist_to_wall < 2:
            wall_penalty = -5 * (2 - dist_to_wall)

        if self.game.game_over:
            done = True
            # Terminal rewards can remain high; adjust if needed.
            reward = 100 if self.game.winner == "Player" else -100
        else:
            # Start with a survival bonus.
            reward = 1

            # Apple Distance Reward:
            # Compute the current Manhattan distance from the player's head to the apple.
            current_distance = manhattan(head, self.game.apple)
            # Reward improvement if the snake gets closer to the apple.
            delta_distance = self.prev_apple_distance - current_distance
            # Scale down the apple reward to prevent risky moves when near walls.
            apple_reward = 1.5 * delta_distance
            reward += apple_reward

            # Optionally, add a small growth bonus if the snake has grown.
            if len(self.game.player_snake) > self.prev_length:
                reward += 20

            # Incorporate the wall proximity penalty.
            reward += wall_penalty

            # Update state-dependent variables.
            self.prev_apple_distance = current_distance
            self.prev_length = len(self.game.player_snake)

        return state, reward, done

# A simple neural network model for DQN.
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# DQNAgent encapsulating the learning algorithm.
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95  # Discount rate
        self.epsilon = 1.0  # Exploration rate (start high)
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995  # Decay factor for exploration rate
        self.learning_rate = 0.001
        self.batch_size = 64
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        # Epsilon-greedy action selection.
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state_tensor = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            act_values = self.model(state_tensor)
        return torch.argmax(act_values).item()
    
    def replay(self):
        # Start training only when we have enough memory.
        if len(self.memory) < self.batch_size:
            return
        
        minibatch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([exp[0] for exp in minibatch]).to(self.device)
        actions = torch.LongTensor([exp[1] for exp in minibatch]).to(self.device)
        rewards = torch.FloatTensor([exp[2] for exp in minibatch]).to(self.device)
        next_states = torch.FloatTensor([exp[3] for exp in minibatch]).to(self.device)
        dones = torch.FloatTensor([float(exp[4]) for exp in minibatch]).to(self.device)
        
        # Current Q-values for the taken actions.
        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        # Next Q-value (using the max over the next state's possible actions)
        next_q_values = self.model(next_states).max(1)[0]
        
        # Compute target Q-values.
        target = rewards + self.gamma * next_q_values * (1 - dones)
        
        loss = self.criterion(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon to reduce exploration gradually.
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Training loop for DQN.
def train_dqn(episodes):
    def get_death_info(env):
        """
        Analyzes the final state of the game to determine why the DQN snake died.
        Returns a string with the likely death cause.
        """
        # If no cells remain (should not normally happen), report that.
        if env.game.player_died_by:
            return env.game.player_died_by
    
        return "Unknown"

    env = SnakeEnv()
    state_size = env.get_state().shape[0]
    action_size = 4  # up, down, left, right
    agent = DQNAgent(state_size, action_size)
    best_snake_length = 0
    player_win_count = 0
    death_stats = {"AI": 0, "Player":0 , "Wall": 0, "Unknown": 0}
    
    # For tracking average rewards over recent episodes.
    recent_rewards = []
    
    for e in range(episodes):
        state = env.reset()
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
            
            # Train the agent after each step.
            agent.replay()
            
            if done:
                best_snake_length = max(len(env.game.player_snake), best_snake_length)
                if env.game.winner == "Player":
                    player_win_count += 1
                    
                # Track recent rewards.
                recent_rewards.append(total_reward)
                if len(recent_rewards) > 200:
                    recent_rewards.pop(0)
                avg_recent_reward = np.mean(recent_rewards)
                death_cause = get_death_info(env)
                death_stats[death_cause] += 1
                # Print training diagnostics every 200 episodes.
                if (e + 1) % 200 == 0:
                    
                    print(f"Episode {e+1}/{episodes} - Steps: {step_count} - Total Reward: {total_reward:.2f} - "
                          f"Epsilon: {agent.epsilon:.2f} - Best Length: {best_snake_length} - "
                          f"Player Wins: {player_win_count} - Death Causes: {death_stats} - "
                          f"Avg Reward (last 200): {avg_recent_reward:.2f}")
                break

if __name__ == '__main__':
    episodes = 4_000  # You can experiment with higher numbers (e.g., 10k+ episodes) for better performance.
    train_dqn(episodes)
