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
MODEL_FILENAME = "_classic_model.pth"  # Base filename

# Global flags for training control
training_thread = None
training_paused = False
training_stop = False
download_model_flag = False

# Global variables to store training status and best episode replay
global_status = {"current_episode": 0, "avg_reward": 0.0, "board_size": None}

# Track best replays & “special” episodes
best_episode_reward = -float('inf')
best_episode_replay = None
special_points = []
global_max_running_avg = 0.0

# Evaluation globals (populated on /start_evaluate)
eval_env = None
eval_agent = None

# ------------------ Classic Training Environment ------------------
class ClassicTrainGame:
    """
    A simplified classic snake game for training.
    The state is a (1, board_size, board_size) grid:
      - Snake cells: 1
      - Apple cell: 3
    """
    def __init__(self, board_size):
        self.board_size = board_size
        self.reset()

    def reset(self):
        self.grid_width = self.board_size
        self.grid_height = self.board_size
        self.snake = [(self.board_size // 2, self.board_size // 2)]
        self.direction = (1, 0)  # Start moving right.
        self.apple = self._generate_apple()
        self.game_over = False
        self.move_count = 0
        self.prev_apple_distance = self.manhattan(self.snake[0], self.apple)
        # Set a maximum moves threshold based on board size.
        if self.board_size == 6:
            self.max_moves = 200
        elif self.board_size == 10:
            self.max_moves = 400
        else:
            self.max_moves = 800

        return self.get_state()

    def _generate_apple(self):
        while True:
            apple = (random.randint(0, self.board_size - 1),
                     random.randint(0, self.board_size - 1))
            if apple not in self.snake:
                return apple

    def get_state(self):
        grid = np.zeros((self.board_size, self.board_size), dtype=np.float32)
        for cell in self.snake:
            grid[cell[0], cell[1]] = 1.0
        grid[self.apple[0], self.apple[1]] = 3.0
        return np.expand_dims(grid, axis=0)  # Shape: (1, board_size, board_size)
    
    def manhattan(self, a, b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1])

    def step(self, action):
        mapping = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}
        new_direction = mapping[action]
        # Prevent illegal reversal if snake length > 1.
        if (self.direction[0] * -1, self.direction[1] * -1) == new_direction:
            new_direction = self.direction
        else:
            self.direction = new_direction

        head = self.snake[0]
        new_head = (head[0] + self.direction[0], head[1] + self.direction[1])
        
        self.move_count += 1
        baseSurvivalReward = 1 - 0.05 * max(0, self.move_count - self.max_moves)
        
        reward = baseSurvivalReward  # Survival reward

        # --- Manhattan distance reward component ---
        # Compute current Manhattan distance to the apple.
        current_distance = self.manhattan(new_head, self.apple)
        # If prev_apple_distance exists, calculate the change in distance.
        if hasattr(self, 'prev_apple_distance'):
            delta_distance = self.prev_apple_distance - current_distance
            # Scale factor for the distance reward (adjust this factor as needed).
            distance_reward = 0.1 * delta_distance
        reward += distance_reward
        # Update the previous distance.
        self.prev_apple_distance = current_distance

        # Check for wall collision.
        if not (0 <= new_head[0] < self.board_size and 0 <= new_head[1] < self.board_size):
            self.game_over = True
            reward = -100
            return self.get_state(), reward, self.game_over

        # Check for self-collision.
        if new_head in self.snake:
            self.game_over = True
            reward = -100
            return self.get_state(), reward, self.game_over

        # Move the snake.
        self.snake.insert(0, new_head)
        if new_head == self.apple:
            reward += 20  # Eating apple bonus.
            self.apple = self._generate_apple()
        else:
            self.snake.pop()

        return self.get_state(), reward, self.game_over
    

# ------------------ ActorCritic Network Definition ------------------
class ActorCritic(nn.Module):
    def __init__(self, grid_width, grid_height, action_size):
        super(ActorCritic, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * grid_width * grid_height, 512)
        self.policy_head = nn.Linear(512, action_size)
        self.value_head = nn.Linear(512, 1)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        return policy_logits, value

# ------------------ PPO Agent Definition ------------------
class PPOAgent:
    def __init__(self, grid_width, grid_height, action_size, lr=0.0003, gamma=0.99,
                 eps_clip=0.2, k_epochs=4, gae_lambda=0.95):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.gae_lambda = gae_lambda
        self.action_size = action_size
        self.device = DEVICE
        self.policy = ActorCritic(grid_width, grid_height, action_size).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
    
    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        logits, value = self.policy(state_tensor)
        probs = torch.softmax(logits, dim=1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        return action.item(), action_logprob.item(), value.item()
    
    def compute_gae(self, rewards, masks, values):
        gae = 0
        returns = []
        values = values + [0]
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step+1] * masks[step] - values[step]
            gae = delta + self.gamma * self.gae_lambda * masks[step] * gae
            returns.insert(0, gae + values[step])
        return returns
    
    def update(self, memory):
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
def train_ppo_classic(board_size, total_episodes):
    global training_paused, training_stop, best_episode_reward, best_episode_replay, global_status, download_model_flag
    global global_status, special_points, global_max_running_avg
    env = ClassicTrainGame(board_size)
    action_size = 4
    agent = PPOAgent(env.board_size, env.board_size, action_size)
    timestep = 0
    memory = {'states': [], 'actions': [], 'logprobs': [], 'rewards': [], 'masks': [], 'values': []}
    running_reward = 0
    episode_num = 0

    while episode_num < total_episodes and not training_stop:
        # If download_model_flag is set, save checkpoint immediately.
        if download_model_flag:
            checkpoint_filename = f"{episode_num}_{board_size}{MODEL_FILENAME}"
            torch.save(agent.policy.state_dict(), checkpoint_filename)
            print(f"Checkpoint saved to {checkpoint_filename}")
            download_model_flag = False

        # Pause check.
        while training_paused and not training_stop:
            time.sleep(1)
        state = env.reset()
        done = False
        ep_reward = 0.0
        episode_replay = []
        # Save initial state snapshot.
        episode_replay.append({
            "snake": env.snake.copy(),
            "apple": env.apple
        })
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
            episode_replay.append({
                "snake": env.snake.copy(),
                "apple": env.apple
            })
            if timestep % 2000 == 0:
                returns = agent.compute_gae(memory['rewards'], memory['masks'], memory['values'])
                memory['returns'] = returns
                agent.update(memory)
                memory = {'states': [], 'actions': [], 'logprobs': [], 'rewards': [], 'masks': [], 'values': []}
                timestep = 0

        # Optionally, update with remaining memory at end-of-episode.
        if len(memory['states']) > 0:
            returns = agent.compute_gae(memory['rewards'], memory['masks'], memory['values'])
            memory['returns'] = returns
            agent.update(memory)
            memory = {'states': [], 'actions': [], 'logprobs': [], 'rewards': [], 'masks': [], 'values': []}
            timestep = 0

        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        episode_num += 1
        global_status["current_episode"] = episode_num
        global_status["avg_reward"] = running_reward
        global_status["board_size"] = board_size
        
        if episode_num % 1_000 == 0:
            print(f"Train Episode {episode_num}/{total_episodes} Reward: {ep_reward:.2f} Avg Reward: {running_reward:.2f}")

        # If episode >= 10k and current reward exceeds previous best, record replay.
        if episode_num >= 10000 and ep_reward > best_episode_reward:
            best_episode_reward = ep_reward
            best_episode_replay = episode_replay
            print(f"New best replay at episode {episode_num} with reward {ep_reward:.2f}")

        # mark special points
        is_special = (ep_reward > running_reward*1.1)
        if is_special:
            special_points.append(episode_num)
            checkpoint_filename = f"{episode_num}_{board_size}{MODEL_FILENAME}"
            torch.save(agent.policy.state_dict(), checkpoint_filename)
            print(f"Saved checkpoint to {checkpoint_filename}")
        elif episode_num > 50_000 and episode_num % 10_000 == 0:
            special_points.append(episode_num)
            checkpoint_filename = f"{episode_num}_{board_size}{MODEL_FILENAME}"
            torch.save(agent.policy.state_dict(), checkpoint_filename)
            print(f"Saved checkpoint to {checkpoint_filename}")

    checkpoint_filename = f"{episode_num}_{board_size}{MODEL_FILENAME}"
    torch.save(agent.policy.state_dict(), checkpoint_filename)
    print("Training completed and final model saved.")

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
    global training_thread, training_stop, training_paused, global_status, best_episode_reward, best_episode_replay, download_model_flag
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
    global download_model_flag
    download_model_flag = True
    return jsonify({"status": "Download requested; checkpoint will be saved shortly."})

@app_train.route("/download_model_file", methods=["GET"])
def download_model_file():
    """
    Download the current model checkpoint file.
    This endpoint expects that the global MODEL_FILENAME exists.
    """
    if not os.path.exists(MODEL_FILENAME):
        return jsonify({"status": "Model file not found."}), 404
    return send_file(MODEL_FILENAME, as_attachment=True)

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

@app_train.route("/get_special_points", methods=["GET"])
def get_special():
    return jsonify({"special_points": special_points})

# ------------------ Evaluation Endpoints ------------------
@app_train.route("/start_evaluate", methods=["POST"])
def start_evaluate():
    """
    Load the {episode}_{board_size}_classic_model.pth,
    pause training, and reset eval_env & eval_agent.
    """
    global training_paused, eval_env, eval_agent
    data = request.get_json()
    ep = int(data.get("episode", -1))
    sz = int(data.get("board_size", global_status["board_size"] or 20))
    path = f"{ep}_{sz}{MODEL_FILENAME}"
    if not os.path.exists(path):
        return jsonify({"error":f"Model file not found: {path}"}),404

    training_paused = True
    eval_env   = ClassicTrainGame(sz)
    eval_agent = PPOAgent(sz, sz, 4)
    eval_agent.policy.load_state_dict(torch.load(path, map_location=DEVICE))
    eval_agent.policy.eval()

    eval_env.reset()
    return jsonify({"status":f"Evaluation started using {path}"})

@app_train.route("/eval_step", methods=["GET"])
def eval_step():
    """
    Perform one step of the eval_env with eval_agent, return snake+apple.
    """
    global eval_env, eval_agent
    if eval_env is None or eval_agent is None:
        return jsonify({"error":"no eval running"}),400

    state = eval_env.get_state()
    action,_,_ = eval_agent.select_action(state)
    _, _, done = eval_env.step(action)

    return jsonify({
        "snake": eval_env.snake,
        "apple": eval_env.apple,
        "game_over": done
    })

@app_train.route("/stop_evaluate", methods=["POST"])
def stop_evaluate():
    """
    Resume training and tear down eval_env/agent.
    """
    global training_paused, eval_env, eval_agent
    training_paused = False
    eval_env = None
    eval_agent = None
    return jsonify({"status":"Evaluation stopped; training resumed."})


if __name__ == '__main__':
    # Run the training Flask app on port 5001.
    app_train.run(host="0.0.0.0", port=5001, debug=True)
