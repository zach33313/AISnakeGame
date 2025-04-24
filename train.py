import random
import threading, time, os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from collections import deque

# Global settings for training
ALLOWED_SIZES = [6, 10, 20]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_BASE_PATH = r"C:\Users\NAME\Downloads\trainmodel"  # Base path for models
MODEL_FILENAME = r"C:\Users\NAME\Downloads\doesnt_exist.pth"  # Use the downloaded model

# Global flags for training control
training_thread = None
training_paused = False
training_stop = False
download_model_flag = False
current_model_path = None  # Store the path of the most recently saved model

# Global variables to store training status and best episode replay
global_status = {
    "current_episode": 0,
    "avg_reward": 0,
    "board_size": None
}
best_episode_reward = -float('inf')
best_episode_replay = None  # Will store a list of game state dictionaries

# ------------------ Classic Training Environment ------------------
class ClassicTrainGame:
    """
    A simplified classic snake game for training.
    The state is a vector of binary values representing:
      - Danger straight ahead, to the right, to the left
      - Current direction (one-hot encoded)
      - Food location relative to snake (up, down, left, right)
    """
    def __init__(self, board_size):
        self.board_size = board_size
        self.reset()

    def reset(self):
        try:
            self.grid_width = self.board_size
            self.grid_height = self.board_size
            
            # Randomize snake starting position
            start_x = random.randint(0, self.board_size - 1)
            start_y = random.randint(0, self.board_size - 1)
            self.snake = [(start_x, start_y)]
            
            # Randomize starting direction
            directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]  # right, down, left, up
            self.direction = random.choice(directions)
            
            # Generate apple in a position different from the snake
            self.apple = self._generate_apple()
            
            self.game_over = False
            self.move_count = 0
            self.prev_apple_distance = self.manhattan(self.snake[0], self.apple)
            # Set a maximum moves threshold based on board size.
            if self.board_size == 6:
                self.max_moves = 200
            elif self.board_size == 10:
                self.max_moves = 1_000
            else:
                self.max_moves = 3_000
    
            return self.get_state()
        except Exception as e:
            print(f"Error in reset: {e}")
            # Return safe default state
            return np.zeros(11, dtype=np.float32)

    def _generate_apple(self):
        max_attempts = 100  # Prevent infinite loop
        attempts = 0
        while attempts < max_attempts:
            attempts += 1
            apple = (random.randint(0, self.board_size - 1),
                     random.randint(0, self.board_size - 1))
            if apple not in self.snake:
                return apple
        
        # Fallback if we can't find a free space after max attempts
        for x in range(self.board_size):
            for y in range(self.board_size):
                if (x, y) not in self.snake:
                    return (x, y)
        
        # Last resort - place on snake tail if snake fills board
        return self.snake[-1]

    def get_state(self):
        """
        Returns a vector of 11 binary features:
        1-3: Danger (straight, right, left)
        4-7: Direction (west, east, north, south)
        8-11: Food direction (west, east, north, south)
        """
        try:
            head = self.snake[0]
            
            # Define the current direction and relative positions
            # Current direction is (dx, dy)
            dx, dy = self.direction
            
            # Map directions to cardinal directions
            is_direction_west = dx == -1
            is_direction_east = dx == 1
            is_direction_north = dy == -1
            is_direction_south = dy == 1
            
            # Define positions for danger detection
            # Get position for straight ahead (current direction)
            point_straight = (head[0] + dx, head[1] + dy)
            
            # Get position for right turn - rotated 90 degrees clockwise
            right_dx, right_dy = -dy, dx
            point_right = (head[0] + right_dx, head[1] + right_dy)
            
            # Get position for left turn - rotated 90 degrees counter-clockwise
            left_dx, left_dy = dy, -dx
            point_left = (head[0] + left_dx, head[1] + left_dy)
            
            # Check for dangers
            danger_straight = self._is_collision(point_straight)
            danger_right = self._is_collision(point_right)
            danger_left = self._is_collision(point_left)
            
            # Food position relative to snake
            food_west = self.apple[0] < head[0]
            food_east = self.apple[0] > head[0]
            food_north = self.apple[1] < head[1]
            food_south = self.apple[1] > head[1]
            
            # Combine all features into one binary vector
            state = [
                danger_straight,
                danger_right,
                danger_left,
                is_direction_west,
                is_direction_east,
                is_direction_north,
                is_direction_south,
                food_west,
                food_east,
                food_north,
                food_south
            ]
            
            # Convert boolean values to binary (0 or 1)
            return np.array(state, dtype=np.float32)
        except Exception as e:
            print(f"Error in get_state: {e}")
            # Return safe default state
            return np.zeros(11, dtype=np.float32)
    
    def _is_collision(self, point):
        """Check if a point would result in collision with wall or snake body"""
        try:
            # Check for wall collision
            if (point[0] < 0 or point[0] >= self.board_size or 
                point[1] < 0 or point[1] >= self.board_size):
                return True
            
            # Check for self collision (except the tail which will move)
            if point in self.snake and point != self.snake[-1]:
                return True
                
            return False
        except Exception as e:
            print(f"Error in _is_collision: {e}")
            # Assume collision if error (fail-safe)
            return True
    
    def step(self, action):
        try:
            # Action: 0=straight, 1=right turn, 2=left turn
            # First, convert to absolute directions based on current orientation
            
            # Validate action
            if not isinstance(action, (int, np.int64, np.int32)) or action < 0 or action > 2:
                print(f"Invalid action: {action}, using default (straight)")
                action = 0  # Default to straight
            
            # Get current direction vector
            dx, dy = self.direction
            
            if action == 0:  # Continue straight
                new_direction = (dx, dy)
            elif action == 1:  # Turn right
                new_direction = (-dy, dx)  # 90 degree clockwise rotation
            elif action == 2:  # Turn left
                new_direction = (dy, -dx)  # 90 degree counter-clockwise rotation
            else:
                # Should never get here due to validation above
                new_direction = (dx, dy)
            
            # Update direction
            self.direction = new_direction
            
            head = self.snake[0]
            new_head = (head[0] + self.direction[0], head[1] + self.direction[1])
            
            self.move_count += 1
            
            # Initialize reward
            reward = 0.01  # Small positive reward for each step (survival bonus)
            
            # Check if the move results in collision
            if self._is_collision(new_head):
                self.game_over = True
                reward = -1.0  # Collision penalty
                return self.get_state(), reward, self.game_over
                
            # Move the snake
            self.snake.insert(0, new_head)
            
            # Apple consumption reward
            if new_head == self.apple:
                # Fixed reward for eating apple
                reward += 1.0
                
                # Bonus for eating apples quickly
                time_bonus = max(0, 1.0 - (self.move_count / (self.board_size * 6)))
                reward += time_bonus
                
                # Generate new apple
                self.apple = self._generate_apple()
                self.move_count = 0
            else:
                # Remove tail if no apple eaten
                self.snake.pop()
                length_factor = len(self.snake) / (self.board_size * self.board_size)   # ∈(0,1]
                free_frac, apple_dist = self.flood_fill_space()
                free_frac /= (self.board_size*self.board_size)
                reward += length_factor * free_frac
               



                # Simple directional guidance using Manhattan distance
                current_distance = self.manhattan(new_head, self.apple)
                
                if apple_dist != float('inf'):
                    reward += length_factor * (apple_dist / (current_distance + 1) - 1)

               
                if hasattr(self, 'prev_apple_distance'):
                    distance_change = self.prev_apple_distance - current_distance
                    # Small reward for moving toward apple, small penalty for moving away
                    reward += 0.05 * distance_change
                
                # Update previous distance
                self.prev_apple_distance = current_distance
                
                # Starvation penalty - strongly penalize not eating for too long
                # Set a fixed limit of 100 moves or board_size*5, whichever is larger
                starvation_limit = max(100, self.board_size * 2)
                if self.move_count > starvation_limit:
                    # Exponential penalty for exceeding starvation limit
                    excess = self.move_count - starvation_limit
                    starvation_penalty = 0.01 * (2 ** min(5, excess / 10))
                    reward -= starvation_penalty
                    
                    # End the game if the snake has gone for extremely long without eating
                    if self.move_count > starvation_limit * 2:
                        self.game_over = True
                        reward -= 1.0  # Additional penalty for extreme starvation
                        return self.get_state(), reward, self.game_over
            
            # Proximity danger penalty - discourage getting close to the snake's body
            # Count how many adjacent cells are occupied by snake body (not including head)
            danger_count = 0
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                adjacent_pos = (new_head[0] + dx, new_head[1] + dy)
                # Check if in bounds
                if (0 <= adjacent_pos[0] < self.board_size and 
                    0 <= adjacent_pos[1] < self.board_size and
                    adjacent_pos in self.snake[1:]):
                    danger_count += 1
            
            # Penalize proximity to own body - stronger penalty when more surrounded
            # This teaches the snake to avoid getting boxed in
            if danger_count > 0:
                reward -= 0.1 * danger_count
    
            return self.get_state(), reward, self.game_over
        except Exception as e:
            print(f"Error in step method: {e}")
            # Return a safe default state with game over flag
            self.game_over = True
            return np.zeros(11, dtype=np.float32), -1.0, True
        
    def manhattan(self, a, b):
        try:
            return abs(a[0]-b[0]) + abs(a[1]-b[1])
        except Exception as e:
            print(f"Error in manhattan: {e}")
            # Return safe default
            return 0
        
    def flood_fill_space(self):
        head = self.snake[0]
        visited = set(self.snake)  # treat body as walls
        q = deque([(head[0], head[1], 0)])  # (x, y, distance)
        visited.add(head)

        count = 0
        apple_dist = None

        while q:
            x, y, dist = q.popleft()
            for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.board_size and 0 <= ny < self.board_size and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    count += 1
                    # record distance when we hit the apple
                    if (nx, ny) == self.apple and apple_dist is None:
                        apple_dist = dist + 1
                    q.append((nx, ny, dist + 1))

        if apple_dist is None:
            apple_dist = float('inf')

        return count, apple_dist




# ------------------ ActorCritic Network Definition ------------------
class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256):
        super(ActorCritic, self).__init__()
        # Simple feed-forward network for binary features
        self.state_size = state_size
        self.action_size = action_size
        
        # Shared layers
        self.shared_layers = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Policy head - predicts action probabilities
        self.policy_head = nn.Linear(hidden_size, action_size)
        
        # Value head - predicts state value
        self.value_head = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        try:
            # Ensure input is a tensor with proper shape
            if not isinstance(x, torch.Tensor):
                x = torch.FloatTensor(x).to(next(self.parameters()).device)
                
            # Add batch dimension if not present
            if x.dim() == 1:
                x = x.unsqueeze(0)
                
            # Ensure the state size is correct
            if x.shape[1] != self.state_size:
                print(f"Warning: Expected state size {self.state_size}, got {x.shape[1]}. Padding state.")
                # Pad or truncate to match expected state size
                if x.shape[1] < self.state_size:
                    padding = torch.zeros(x.shape[0], self.state_size - x.shape[1], device=x.device)
                    x = torch.cat([x, padding], dim=1)
                else:
                    x = x[:, :self.state_size]
                
            x = self.shared_layers(x)
            policy_logits = self.policy_head(x)
            value = self.value_head(x)
            return policy_logits, value
        except Exception as e:
            print(f"Error in ActorCritic forward pass: {e}")
            # Return safe defaults
            device = next(self.parameters()).device
            return torch.zeros(1, self.action_size, device=device), torch.zeros(1, 1, device=device)

# ------------------ PPO Agent Definition ------------------
class PPOAgent:
    def __init__(self, state_size=11, action_size=3, lr=0.0001, gamma=0.99,
                 eps_clip=0.1, k_epochs=8, gae_lambda=0.95):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.gae_lambda = gae_lambda
        self.action_size = action_size
        self.state_size = state_size
        self.device = DEVICE
        self.policy = ActorCritic(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        # Add entropy coefficient to encourage exploration
        self.entropy_coeff = 0.01
    
    def select_action(self, state):
        try:
            with torch.no_grad():
                # Validate state
                if not isinstance(state, np.ndarray):
                    print(f"Warning: state is not a numpy array, converting. Type: {type(state)}")
                    state = np.array(state, dtype=np.float32)
                
                # Check for NaN values
                if np.isnan(state).any():
                    print("Warning: NaN values in state, replacing with zeros")
                    state = np.nan_to_num(state, nan=0.0)
                
                # Ensure state has right size
                if state.size != self.state_size:
                    print(f"Warning: state size mismatch. Expected {self.state_size}, got {state.size}. Resizing.")
                    if state.size < self.state_size:
                        # Pad with zeros
                        new_state = np.zeros(self.state_size, dtype=np.float32)
                        new_state[:state.size] = state
                        state = new_state
                    else:
                        # Truncate
                        state = state[:self.state_size]
                
                state_tensor = torch.FloatTensor(state).to(self.device)
                logits, value = self.policy(state_tensor)
                
                # Clip logits for numerical stability
                logits = torch.clamp(logits, -20.0, 20.0)
                probs = torch.softmax(logits, dim=-1)
                
                # Add small epsilon to avoid zero probabilities
                epsilon = 1e-6
                probs = probs + epsilon
                probs = probs / probs.sum(dim=-1, keepdim=True)  # Renormalize
                
                # Check for valid probabilities
                if torch.isnan(probs).any() or torch.isinf(probs).any():
                    # Fall back to uniform distribution if NaN detected
                    print("Warning: NaN in action probabilities, using uniform distribution")
                    probs = torch.ones_like(probs) / probs.size(-1)
                    
                # Safely sample action
                try:
                    dist = torch.distributions.Categorical(probs)
                    action = dist.sample()
                    action_logprob = dist.log_prob(action)
                except Exception as e:
                    print(f"Error sampling action: {e}, using random action")
                    action = torch.randint(0, self.action_size, (1,), device=self.device)
                    action_logprob = torch.tensor([-1.0], device=self.device)  # Dummy value
                
                # Convert to Python primitives for serialization
                return int(action.item()), float(action_logprob.item()), float(value.item())
        except Exception as e:
            print(f"Critical error in select_action: {e}, returning random action")
            return random.randint(0, self.action_size-1), 0.0, 0.0
    
    def compute_gae(self, rewards, masks, values):
        gae = 0
        returns = []
        # Add extra zero as last value
        values = values + [0]
        
        # Convert to numpy for more stable computation
        rewards = np.array(rewards, dtype=np.float32)
        masks = np.array(masks, dtype=np.float32)
        values = np.array(values, dtype=np.float32)
        
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step+1] * masks[step] - values[step]
            # Clamp delta for stability
            delta = np.clip(delta, -10.0, 10.0)
            gae = delta + self.gamma * self.gae_lambda * masks[step] * gae
            # Clamp gae for stability
            gae = np.clip(gae, -10.0, 10.0)
            returns.insert(0, gae + values[step])
        
        # Convert back to list and check for NaN
        returns = [float(x) for x in returns]
        if any(np.isnan(returns)) or any(np.isinf(returns)):
            print("Warning: NaN or Inf values in returns, using simpler returns")
            # Fall back to simpler return calculation
            returns = []
            R = 0
            for r, m in zip(reversed(rewards), reversed(masks)):
                R = r + self.gamma * R * m
                returns.insert(0, R)
                
        return returns
    
    def update(self, memory):
        states = torch.FloatTensor(np.array(memory['states'])).to(self.device)
        actions = torch.LongTensor(memory['actions']).to(self.device)
        old_logprobs = torch.FloatTensor(memory['logprobs']).to(self.device)
        returns = torch.FloatTensor(memory['returns']).to(self.device)
        advantages = returns - torch.FloatTensor(memory['values']).to(self.device)
        
        # Safer advantage normalization with check for non-zero std
        if advantages.shape[0] > 1 and advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        for _ in range(self.k_epochs):
            logits, values = self.policy(states)
            
            # Apply a softer softmax with temperature for numerical stability
            temperature = 1.0
            scaled_logits = logits / temperature
            # Clip logits to prevent extreme values
            scaled_logits = torch.clamp(scaled_logits, -20.0, 20.0)
            probs = torch.softmax(scaled_logits, dim=1)
            
            # Add small epsilon to avoid zero probabilities
            epsilon = 1e-6
            probs = probs + epsilon
            probs = probs / probs.sum(dim=1, keepdim=True)  # Renormalize
            
            # Check for NaN values
            if torch.isnan(probs).any() or torch.isinf(probs).any():
                print("Warning: NaN or Inf values in probabilities, skipping update")
                return
                
            dist = torch.distributions.Categorical(probs)
            new_logprobs = dist.log_prob(actions)
            
            # Use safer exp with clipping
            ratio = torch.exp(torch.clamp(new_logprobs - old_logprobs, -20.0, 20.0))
            
            # Clip ratio for stability
            ratio = torch.clamp(ratio, 0.0, 10.0)
            
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            # Calculate policy loss, value loss, and entropy loss separately
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Fix value loss dimension mismatch
            value_loss = 0.5 * ((values.squeeze() - returns.squeeze()) ** 2).mean()
            
            entropy_loss = -self.entropy_coeff * dist.entropy().mean()
            
            # Total loss
            loss = policy_loss + value_loss + entropy_loss
            
            # Check for NaN in loss
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                print("Warning: NaN or Inf values in loss, skipping update")
                return
            
            self.optimizer.zero_grad()
            loss.backward()
            
            # More aggressive gradient clipping
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()

def adapt_model_weights(old_state_dict):
    """Adapt weights from a 1-channel model to a 2-channel model by duplicating the first conv layer"""
    new_state_dict = old_state_dict.copy()
    
    # Get the first conv layer weights
    old_conv1_weight = old_state_dict['conv1.weight']  # Shape: [32, 1, 3, 3]
    
    # Create new weights by duplicating the channel dimension
    # The first channel remains the same, the second channel is a copy
    new_conv1_weight = torch.zeros((32, 2, 3, 3), device=old_conv1_weight.device)
    new_conv1_weight[:, 0, :, :] = old_conv1_weight[:, 0, :, :]  # Copy to first channel
    new_conv1_weight[:, 1, :, :] = old_conv1_weight[:, 0, :, :]  # Copy to second channel
    
    # Update the state dict with the new weights
    new_state_dict['conv1.weight'] = new_conv1_weight
    
    return new_state_dict

# ------------------ PPO Training Loop ------------------
def train_ppo_classic(board_size, total_episodes):
    global training_paused, training_stop, best_episode_reward, best_episode_replay, global_status, download_model_flag, current_model_path
    
    # Add exception handling for the entire training loop
    try:
        env = ClassicTrainGame(board_size)
        
        # With binary state, we have 11 state features and 3 actions (straight, right, left)
        state_size = 11
        action_size = 3
        
        agent = PPOAgent(state_size=state_size, action_size=action_size)
        
        # Create backup directory if it doesn't exist
        backup_dir = os.path.join(MODEL_BASE_PATH, "backups")
        if not os.path.exists(backup_dir):
            os.makedirs(backup_dir)
        
        # Load existing model if available
        if os.path.exists(MODEL_FILENAME):
            try:
                # Try to load the model directly
                agent.policy.load_state_dict(torch.load(MODEL_FILENAME, map_location=DEVICE))
                print(f"Loaded model from {MODEL_FILENAME}")
                # Set this as current model path
                current_model_path = MODEL_FILENAME
            except RuntimeError as e:
                # For errors, just report and continue with a fresh model
                print(f"Error loading model (expected with architecture change): {e}")
                print("Starting with a fresh model")
                # Save a new initial model to use as the current path
                initial_model_path = os.path.join(MODEL_BASE_PATH, f"0_{board_size}_classic_modelFF.pth")
                torch.save(agent.policy.state_dict(), initial_model_path)
                current_model_path = initial_model_path
        
        timestep = 0
        memory = {'states': [], 'actions': [], 'logprobs': [], 'rewards': [], 'masks': [], 'values': []}
        running_reward = 0
        episode_num = 0
        
        # Performance tracking
        reward_history = []
        avg_length_history = []
        apples_eaten_history = []
        lr_history = []
        no_improvement_count = 0
        best_running_reward = -float('inf')
        
        # Learning rate with warmup
        initial_lr = 0.00005
        peak_lr = 0.0001
        min_lr = 0.00001
        warmup_episodes = 5000
        
        # Set initial learning rate
        for param_group in agent.optimizer.param_groups:
            param_group['lr'] = initial_lr

        # Add memory limit counter
        memory_reset_counter = 0
        
        while episode_num < total_episodes and not training_stop:
            try:
                # Memory safety - periodically clear memory and save model to prevent memory issues
                memory_reset_counter += 1
                if memory_reset_counter >= 1000:  # Clear every 1000 episodes
                    print(f"Performing periodic memory cleanup at episode {episode_num}")
                    # Clear Python memory
                    memory = {'states': [], 'actions': [], 'logprobs': [], 'rewards': [], 'masks': [], 'values': []}
                    
                    # Force garbage collection
                    import gc
                    gc.collect()
                    
                    # Empty CUDA cache if using GPU
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # Save backup model
                    backup_path = os.path.join(backup_dir, f"backup_{episode_num}_{board_size}_modelFF.pth")
                    torch.save(agent.policy.state_dict(), backup_path)
                    print(f"Memory cleanup complete. Backup saved to {backup_path}")
                    memory_reset_counter = 0
                
                # If download_model_flag is set, save checkpoint immediately.
                if download_model_flag:
                    checkpoint_filename = os.path.join(MODEL_BASE_PATH, f"{episode_num}_{board_size}_classic_modelFF.pth")
                    torch.save(agent.policy.state_dict(), checkpoint_filename)
                    print(f"Checkpoint saved to {checkpoint_filename}")
                    current_model_path = checkpoint_filename
                    download_model_flag = False
                    
                # Pause check.
                while training_paused and not training_stop:
                    time.sleep(1)
                state = env.reset()
                done = False
                ep_reward = 0
                episode_replay = []
                apples_eaten = 0
                # Save initial state snapshot.
                episode_replay.append({
                    "snake": env.snake.copy(),
                    "apple": env.apple
                })
                
                # Print debug info every 100 episodes
                if episode_num % 100 == 0:
                    print(f"Starting episode {episode_num}, state shape: {state.shape if hasattr(state, 'shape') else 'unknown'}")
                
                step_count = 0
                max_steps_per_episode = max(env.board_size * 100, 1000)  # Prevent infinite episodes
                
                while not done and step_count < max_steps_per_episode:
                    try:
                        timestep += 1
                        step_count += 1
                        
                        # Get action
                        action, logprob, value = agent.select_action(state)
                        
                        # Debug every 1000 episodes to track state and actions
                        if episode_num % 1000 == 0 and step_count <= 3:
                            print(f"Episode {episode_num}, Step {step_count}")
                            print(f"State: {state}")
                            print(f"Action: {action}, LogProb: {logprob}, Value: {value}")
                        
                        # Store in memory
                        memory['states'].append(state)
                        memory['actions'].append(action)
                        memory['logprobs'].append(logprob)
                        memory['values'].append(value)
                        
                        # Take step
                        next_state, reward, done = env.step(action)
                        
                        memory['rewards'].append(reward)
                        mask = 0 if done else 1
                        memory['masks'].append(mask)
                        state = next_state
                        ep_reward += reward
                        
                        # Track apples eaten
                        if len(env.snake) > apples_eaten + 1:
                            apples_eaten = len(env.snake) - 1
                            
                        episode_replay.append({
                            "snake": env.snake.copy(),
                            "apple": env.apple
                        })
                        
                        # Check for NaN in state
                        if np.isnan(state).any():
                            print(f"Warning: NaN detected in state at episode {episode_num}, step {step_count}")
                            # Reset NaN values to 0 to allow training to continue
                            state = np.nan_to_num(state, nan=0.0)
                        
                        # Limit memory size for very long episodes
                        if len(memory['states']) >= 10000:
                            print(f"Memory size limit reached at episode {episode_num}, performing early update")
                            try:
                                returns = agent.compute_gae(memory['rewards'], memory['masks'], memory['values'])
                                memory['returns'] = returns
                                agent.update(memory)
                            except Exception as e:
                                print(f"Error during forced memory update: {e}")
                            finally:
                                memory = {'states': [], 'actions': [], 'logprobs': [], 'rewards': [], 'masks': [], 'values': []}
                                timestep = 0
                        
                        elif timestep % 2000 == 0:
                            # Perform policy update
                            try:
                                returns = agent.compute_gae(memory['rewards'], memory['masks'], memory['values'])
                                memory['returns'] = returns
                                agent.update(memory)
                                memory = {'states': [], 'actions': [], 'logprobs': [], 'rewards': [], 'masks': [], 'values': []}
                                timestep = 0
                            except Exception as e:
                                print(f"Error during policy update: {e}")
                                # Clear memory and continue
                                memory = {'states': [], 'actions': [], 'logprobs': [], 'rewards': [], 'masks': [], 'values': []}
                                timestep = 0
                    
                    except Exception as e:
                        print(f"Error during episode step: {e}")
                        done = True  # End the episode if there's an error
                        memory = {'states': [], 'actions': [], 'logprobs': [], 'rewards': [], 'masks': [], 'values': []}
                        timestep = 0
                
                # Safety check for max steps reached
                if step_count >= max_steps_per_episode:
                    print(f"Episode {episode_num} reached max step limit ({max_steps_per_episode}), forcing termination")
                    done = True

                # Optionally, update with remaining memory at end-of-episode.
                if len(memory['states']) > 0:
                    try:
                        returns = agent.compute_gae(memory['rewards'], memory['masks'], memory['values'])
                        memory['returns'] = returns
                        agent.update(memory)
                        memory = {'states': [], 'actions': [], 'logprobs': [], 'rewards': [], 'masks': [], 'values': []}
                        timestep = 0
                    except Exception as e:
                        print(f"Error during end-of-episode update: {e}")
                        memory = {'states': [], 'actions': [], 'logprobs': [], 'rewards': [], 'masks': [], 'values': []}
                        timestep = 0

                # Track training metrics
                reward_history.append(ep_reward)
                avg_length_history.append(len(env.snake))
                apples_eaten_history.append(apples_eaten)
                curr_lr = agent.optimizer.param_groups[0]['lr']
                lr_history.append(curr_lr)
                
                # Learning rate warmup
                if episode_num < warmup_episodes:
                    # Linearly increase learning rate during warmup
                    lr = initial_lr + (peak_lr - initial_lr) * (episode_num / warmup_episodes)
                    for param_group in agent.optimizer.param_groups:
                        param_group['lr'] = lr
                
                # Check for NaN or infinite rewards
                if np.isnan(ep_reward) or np.isinf(ep_reward):
                    print(f"Warning: Episode {episode_num} had NaN or Inf reward. Using 0 instead.")
                    running_reward = running_reward  # Keep previous value
                    ep_reward = 0  # Reset reward to prevent issues later
                else:
                    running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
                
                episode_num += 1
                global_status["current_episode"] = episode_num
                global_status["avg_reward"] = running_reward
                global_status["board_size"] = board_size
                
                # Adaptive learning rate - reduce if no improvement
                if running_reward > best_running_reward:
                    best_running_reward = running_reward
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                    
                # Reduce learning rate after significant lack of improvement
                if no_improvement_count >= 5000:
                    for param_group in agent.optimizer.param_groups:
                        param_group['lr'] = max(param_group['lr'] * 0.5, min_lr)
                    print(f"Reduced learning rate to {agent.optimizer.param_groups[0]['lr']}")
                    no_improvement_count = 0
                        
                if episode_num % 100 == 0:
                    print(f"Train Episode {episode_num}/{total_episodes} Reward: {ep_reward:.2f} Avg Reward: {running_reward:.2f}")
                    print(f"Apples eaten: {apples_eaten}, Snake Length: {len(env.snake)}, LR: {curr_lr:.6f}")
                    
                # Save checkpoint every 500 episodes for the new architecture
                if episode_num % 500 == 0:
                    checkpoint_filename = os.path.join(MODEL_BASE_PATH, f"{episode_num}_{board_size}_classic_modelFF.pth")
                    torch.save(agent.policy.state_dict(), checkpoint_filename)
                    print(f"Saved checkpoint to {checkpoint_filename}")
                    current_model_path = checkpoint_filename

                # If episode >= 1000 and current reward exceeds previous best, record replay.
                if episode_num >= 1000 and ep_reward > best_episode_reward:
                    best_episode_reward = ep_reward
                    best_episode_replay = episode_replay
                    print(f"New best replay at episode {episode_num} with reward {ep_reward:.2f}")
            
            except Exception as e:
                print(f"Error during episode {episode_num}: {e}")
                # Recover and continue with next episode
                episode_num += 1
                memory = {'states': [], 'actions': [], 'logprobs': [], 'rewards': [], 'masks': [], 'values': []}
                timestep = 0

        # Save final checkpoint.
        checkpoint_filename = os.path.join(MODEL_BASE_PATH, f"{episode_num}_{board_size}_classic_modelFF.pth")
        torch.save(agent.policy.state_dict(), checkpoint_filename)
        current_model_path = checkpoint_filename
        print("Training completed and final model saved.")
        
    except Exception as e:
        # Catch any unexpected errors in the training process
        print(f"Critical error in training: {e}")
        # Try to save the model in case of crash
        if 'agent' in locals() and 'episode_num' in locals():
            emergency_save = os.path.join(MODEL_BASE_PATH, f"emergency_{episode_num}_{board_size}_modelFF.pth")
            try:
                torch.save(agent.policy.state_dict(), emergency_save)
                print(f"Emergency model saved to {emergency_save}")
                current_model_path = emergency_save
            except Exception as save_error:
                print(f"Could not save emergency model: {save_error}")
        return

# ------------------ Flask App for Training Control ------------------
app_train = Flask(__name__)
CORS(app_train, resources={r"/*": {"origins": "*"}})

@app_train.route("/start_train", methods=["POST"])
def start_train():
    """
    Start PPO training for classic snake.
    Expects JSON with:
      - "size": board size (6, 10, or 20)
      - "episodes": total training episodes (default 1000)
    """
    global training_thread, training_stop, training_paused, global_status, best_episode_reward, best_episode_replay, download_model_flag, current_model_path
    if training_thread is not None and training_thread.is_alive():
        return jsonify({"status": "Training already in progress."}), 400
    data = request.get_json()
    try:
        board_size = int(data.get("size", 20))
        if board_size not in ALLOWED_SIZES:
            return jsonify({"status": "Invalid board size. Must be 6, 10, or 20."}), 400
        total_episodes = int(data.get("episodes", 1000))
    except Exception as e:
        return jsonify({"status": "Invalid parameters", "error": str(e)}), 400
    training_stop = False
    training_paused = False
    global_status = {"current_episode": 0, "avg_reward": 0, "board_size": board_size}
    best_episode_reward = -float('inf')
    best_episode_replay = None
    training_thread = threading.Thread(target=train_ppo_classic, args=(board_size, total_episodes))
    training_thread.start()
    return jsonify({"status": f"Training started on {board_size}x{board_size} board for {total_episodes} episodes."})

@app_train.route("/pause_train", methods=["POST"])
def pause_train():
    """
    Pause or resume training.
    Expects JSON with "pause": true to pause, false to resume.
    """
    global training_paused
    data = request.get_json()
    pause_flag = bool(data.get("pause", True))
    training_paused = pause_flag
    status = "paused" if training_paused else "resumed"
    return jsonify({"status": f"Training {status}."})

@app_train.route("/stop_train", methods=["POST"])
def stop_train():
    """
    Stop training gracefully.
    """
    global training_stop
    training_stop = True
    return jsonify({"status": "Training stop requested."})

@app_train.route("/download_model", methods=["GET"])
def download_model():
    """
    Trigger a model checkpoint save; once saved, subsequent calls to download endpoint
    (using /download_model_file) can return the file.
    """
    global download_model_flag, current_model_path, global_status
    
    # Create a model file name based on current episode and board size
    episode_num = global_status["current_episode"]
    board_size = global_status["board_size"]
    
    if episode_num > 0 and board_size is not None:
        # Set the flag to save on next training iteration
        download_model_flag = True
        return jsonify({
            "status": f"Download requested; checkpoint for episode {episode_num} on board size {board_size} will be saved shortly."
        })
    else:
        return jsonify({"status": "No active training session found."}), 400

@app_train.route("/download_model_file", methods=["GET"])
def download_model_file():
    """
    Download the current model checkpoint file.
    This endpoint expects that the global MODEL_FILENAME exists.
    """
    if not os.path.exists(current_model_path):
        return jsonify({"status": "Model file not found."}), 404
    return send_file(current_model_path, as_attachment=True)

@app_train.route("/get_best_replay", methods=["GET"])
def get_best_replay():
    """
    Return the best episode's replay (the sequence of game state snapshots) if available.
    """
    global best_episode_replay, best_episode_reward
    if best_episode_replay is None:
        return jsonify({"status": "No best replay available yet."}), 404
    return jsonify({
        "best_reward": best_episode_reward,
        "replay": best_episode_replay
    })

@app_train.route("/get_training_status", methods=["GET"])
def get_training_status():
    """
    Returns current training status: current episode, average reward, and board size.
    """
    return jsonify(global_status)


if __name__ == '__main__':
    # Run the training Flask app on port 5001.
    app_train.run(host="0.0.0.0", port=5001, debug=True)
