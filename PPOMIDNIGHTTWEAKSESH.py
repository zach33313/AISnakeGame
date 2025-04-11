import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from app import Game, in_bounds, get_neighbors, manhattan  

# ------------------ Environment Wrapper ------------------

class SnakeEnv:
    def __init__(self):
        self.game = Game()
        self.game.reset(mode="versus")
    
    def reset(self):
        # Reset the game in versus mode and initialize helper variables.
        self.game.reset(mode="versus")
        self.prev_apple_distance = manhattan(self.game.player_snake[0], self.game.apple)
        self.prev_length = len(self.game.player_snake)
        return self.get_state()
    
    def get_state(self):
        # Create a grid representation with shape (1, GRID_WIDTH, GRID_HEIGHT)
        grid = np.zeros((self.game.grid_width, self.game.grid_height), dtype=np.float32)
        # Mark the player snake with 1.
        for cell in self.game.player_snake:
            grid[cell[0], cell[1]] = 1.0
        # Mark the opponent (A*) snake with 2.
        for cell in self.game.ai_snake:
            grid[cell[0], cell[1]] = 2.0
        # Mark the apple with 3.
        grid[self.game.apple[0], self.game.apple[1]] = 3.0
        return np.expand_dims(grid, axis=0)  # shape: (1, grid_width, grid_height)
    
    def step(self, action):
        # Map integer action to a direction.
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
        
        # Create a simple reward: base survival, apple-distance bonus, and wall penalty.
        head = self.game.player_snake[0]
        dist_to_wall = min(head[0], self.game.grid_width - 1 - head[0],
                           head[1], self.game.grid_height - 1 - head[1])
        wall_penalty = -5 * (2 - dist_to_wall) if dist_to_wall < 2 else 0
        
        if self.game.game_over:
            done = True
            reward = 100 if self.game.winner == "Player" else -100
        else:
            reward = 1  # Survival reward.
            current_distance = manhattan(head, self.game.apple)
            delta_distance = self.prev_apple_distance - current_distance
            apple_reward = 1.5 * delta_distance
            reward += apple_reward
            # Bonus for growing (i.e. eating an apple).
            if len(self.game.player_snake) > self.prev_length:
                reward += 20
            reward += wall_penalty
            self.prev_apple_distance = current_distance
            self.prev_length = len(self.game.player_snake)
        
        return state, reward, done

# ------------------ Actor-Critic Network for PPO ------------------

class ActorCritic(nn.Module):
    def __init__(self, grid_width, grid_height, action_size):
        super(ActorCritic, self).__init__()
        # Two convolutional layers to capture spatial features.
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # Fully-connected layer: note that conv layers preserve spatial dimensions.
        self.fc1 = nn.Linear(64 * grid_width * grid_height, 512)
        # Policy and value heads.
        self.policy_head = nn.Linear(512, action_size)
        self.value_head = nn.Linear(512, 1)
    
    def forward(self, x):
        # x expected shape: (batch_size, 1, grid_width, grid_height)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        return policy_logits, value

# ------------------ PPO Agent ------------------

class PPOAgent:
    def __init__(self, grid_width, grid_height, action_size, lr=0.0003, gamma=0.99,
                 eps_clip=0.2, k_epochs=4, gae_lambda=0.95):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.gae_lambda = gae_lambda
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = ActorCritic(grid_width, grid_height, action_size).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
    
    def select_action(self, state):
        # Add batch dimension and convert state to tensor.
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        logits, value = self.policy(state_tensor)
        probs = torch.softmax(logits, dim=1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        return action.item(), action_logprob.item(), value.item()
    
    def compute_gae(self, rewards, masks, values):
        # Compute Generalized Advantage Estimation (GAE).
        gae = 0
        returns = []
        values = values + [0]  # Append a terminal zero value.
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step+1] * masks[step] - values[step]
            gae = delta + self.gamma * self.gae_lambda * masks[step] * gae
            returns.insert(0, gae + values[step])
        return returns
    
    def update(self, memory):
        # Convert collected memory to tensors.
        states = torch.FloatTensor(np.array(memory['states'])).to(self.device)
        actions = torch.LongTensor(memory['actions']).to(self.device)
        old_logprobs = torch.FloatTensor(memory['logprobs']).to(self.device)
        returns = torch.FloatTensor(memory['returns']).to(self.device)
        advantages = returns - torch.FloatTensor(memory['values']).to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        for _ in range(self.k_epochs):
            logits, values = self.policy(states)
            probs = torch.softmax(logits, dim=1)
            dist = torch.distributions.Categorical(probs)
            new_logprobs = dist.log_prob(actions)
            ratio = torch.exp(new_logprobs - old_logprobs)
            
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2).mean() + 0.5 * nn.MSELoss()(values.squeeze(), returns) - 0.01 * dist.entropy().mean()
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

# ------------------ PPO Training Loop ------------------

def train_ppo(episodes, update_timestep=2000):
    env = SnakeEnv()
    grid_width = env.game.grid_width
    grid_height = env.game.grid_height
    action_size = 4  # Up, Down, Left, Right.
    agent = PPOAgent(grid_width, grid_height, action_size)
    
    timestep = 0
    memory = {'states': [], 'actions': [], 'logprobs': [], 'rewards': [], 'masks': [], 'values': []}
    running_reward = 0
    
    for episode in range(episodes):
        state = env.reset()  # shape: (1, grid_width, grid_height)
        done = False
        ep_reward = 0
        
        while not done:
            timestep += 1
            action, logprob, value = agent.select_action(state)
            memory['states'].append(state)
            memory['actions'].append(action)
            memory['logprobs'].append(logprob)
            memory['values'].append(value)
            
            next_state, reward, done = env.step(action)
            memory['rewards'].append(reward)
            mask = 0 if done else 1
            memory['masks'].append(mask)
            
            state = next_state
            ep_reward += reward
            
            # If enough timesteps have been collected, update the policy.
            if timestep % update_timestep == 0:
                returns = agent.compute_gae(memory['rewards'], memory['masks'], memory['values'])
                memory['returns'] = returns
                agent.update(memory)
                memory = {'states': [], 'actions': [], 'logprobs': [], 'rewards': [], 'masks': [], 'values': []}
                timestep = 0
        
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode+1}\tReward: {ep_reward:.2f}\tAverage Reward: {running_reward:.2f}")
        
        # Optionally save the model every few episodes.
        if (episode + 1) % 500 == 0:
            torch.save(agent.policy.state_dict(), f"ppo_snake_{episode+1}.pth")

if __name__ == '__main__':
    train_ppo(episodes=4000)
