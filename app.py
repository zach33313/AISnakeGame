import eventlet
eventlet.monkey_patch()

from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
import threading, time, random, heapq
from flask_socketio import SocketIO, emit
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# ------------------ Global Definitions ------------------
GRID_WIDTH = 20
GRID_HEIGHT = 20
MOVE_INTERVAL = 0.2  # seconds between moves

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------ Utility Functions ------------------
def in_bounds(pos):
    x, y = pos
    return 0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT

def get_neighbors(pos):
    x, y = pos
    neighbors = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
    return [n for n in neighbors if in_bounds(n)]

def manhattan(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

# ------------------ Model Definitions ------------------

class ActorCritic(nn.Module):
    def __init__(self, grid_width, grid_height, action_size):
        super(ActorCritic, self).__init__()
        # Two convolutional layers to capture spatial patterns.
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # Fully-connected layer; convolutional layers preserve dimensions (using padding).
        self.fc1 = nn.Linear(64 * grid_width * grid_height, 512)
        # Separate heads for policy and value.
        self.policy_head = nn.Linear(512, action_size)
        self.value_head = nn.Linear(512, 1)
    
    def forward(self, x):
        # x is expected to be of shape (batch_size, 1, grid_width, grid_height)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        return policy_logits, value

def get_ai_observation(game):
    # Build a single-channel grid: 1=AI snake, 2=player snake, 3=apple
    w, h = game.grid_width, game.grid_height
    grid = np.zeros((w, h), dtype=np.float32)
    for cell in game.ai_snake:
        grid[cell[0], cell[1]] = 1.0
    for cell in game.player_snake:
        grid[cell[0], cell[1]] = 2.0
    # pick the right apple for classic vs versus
    apple = getattr(game, 'ai_apple', getattr(game, 'apple', None))
    if apple:
        grid[apple[0], apple[1]] = 3.0
    tensor = torch.tensor(grid, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return tensor.to(DEVICE)

# Load all three PPO models into a dict keyed by board size
ppo_models = {}
for size, fname in [(6, "1000000_6_classic_model.pth"),
                    (10, "2000000_10_classic_model.pth")]:
                    #(20, "ppo_snake_100000.pth")]
    m = ActorCritic(size, size, 4).to(DEVICE)
    m.load_state_dict(torch.load(fname, map_location=DEVICE))
    m.eval()
    ppo_models[size] = m
print("Loaded PPO models for sizes:", list(ppo_models.keys()))



# ------------------ Game Backend ------------------

class Game:
    def __init__(self):
        self.grid_width = GRID_WIDTH
        self.grid_height = GRID_HEIGHT
        self.mode = "classic"
        self.ai_model = "A_STAR"
        self.mode = "classic"
        self.reset()

    def reset(self, mode="classic", board_size=None, ai_model="A_STAR"):
        if board_size:
            global GRID_WIDTH, GRID_HEIGHT
            GRID_WIDTH = GRID_HEIGHT = board_size
            self.grid_width = self.grid_height = board_size
        self.ai_model = ai_model  # "A_STAR" or "PPO"
        self.mode = mode
        if mode == "versus":
            # In versus mode both snakes share one apple.
            self.player_snake = [(self.grid_width // 2, self.grid_height // 2)]
            self.ai_snake = [(self.grid_width // 2, self.grid_height // 2 - 3)]
            self.player_direction = (1, 0)  # player starts moving right
            self.ai_direction = (1, 0)
            self.apple = self._generate_apple(self.player_snake, self.ai_snake)
            # In versus mode death is immediate upon enemy contact.
            self.player_dead = False
            self.ai_dead = False
            # Reset game-over status.
            self.game_over = False
            self.winner = None
            # Initialize cached path for the AI so we don't recalc every move.
            self.ai_path = []
            self.player_died_by = ""
            self.direction_queue = []
        else:
            # Initialize player and AI snakes at different starting positions.
            self.player_snake = [(self.grid_width // 2, self.grid_height // 2)]
            self.ai_snake = [(self.grid_width // 2, self.grid_height // 2 - 3)]
            self.player_direction = (1, 0)  # player starts moving right
            self.ai_direction = (1, 0)      # initial AI direction (will be updated)
            self.place_player_apple()
            self.place_ai_apple()
            # Reset death flags.
            self.player_dead = False
            self.ai_dead = False
            self.game_over = False
            self.winner = None
            self.direction_queue = []

    def _generate_apple(self, *snake_lists):
        while True:
            apple = (random.randint(0, self.grid_width - 1), random.randint(0, self.grid_height - 1))
            collision = any(apple in snake for snake in snake_lists)
            if not collision:
                return apple

    def place_player_apple(self):
        while True:
            apple = (random.randint(0, self.grid_width - 1), random.randint(0, self.grid_height - 1))
            if apple not in self.player_snake and apple not in self.ai_snake:
                self.player_apple = apple
                break

    def place_ai_apple(self):
        while True:
            apple = (random.randint(0, self.grid_width - 1), random.randint(0, self.grid_height - 1))
            if apple not in self.player_snake and apple not in self.ai_snake:
                self.ai_apple = apple
                break

    def change_player_direction(self, new_direction):
        if MOVE_INTERVAL == 0:
            # In training mode, update immediately.
            if (self.player_direction[0] * -1, self.player_direction[1] * -1) == new_direction:
                return
            self.player_direction = new_direction
        else:
            # Enqueue the new direction.
            self.direction_queue.append(new_direction)
    def process_direction_queue(self):
        """
        Process the queue of direction commands, applying the first valid one.
        A command is valid if it is not directly opposite to the current direction.
        """
        while self.direction_queue:
            next_direction = self.direction_queue.pop(0)
            if (self.player_direction[0] * -1, self.player_direction[1] * -1) != next_direction:
                self.player_direction = next_direction
                break
    
    # ------------------ New AI Direction using PPO Model ------------------
    def compute_ai_direction_PPO(self):
        model = ppo_models.get(self.grid_width, ppo_models[self.grid_height])
        obs = get_ai_observation(self)
        print("PPO obs shape:", obs.shape)
        with torch.no_grad():
            logits, _ = model(obs)
            probs = F.softmax(logits, dim=1)
            print("PPO probs:", probs.cpu().numpy())
            action = torch.argmax(probs, dim=1).item()
            print("PPO action:", action)
        return {0:(0,-1), 1:(0,1), 2:(-1,0), 3:(1,0)}[action]



    def compute_ai_direction_astar(self):
        # Standard A* search for classic mode (unchanged)
        if self.mode == "classic":
            start = self.ai_snake[0]
            goal = self.ai_apple
            obstacles = set(self.ai_snake)
            frontier = []
            heapq.heappush(frontier, (0, start))
            came_from = {start: None}
            cost_so_far = {start: 0}
            found = False
            while frontier:
                _, current = heapq.heappop(frontier)
                if current == goal:
                    found = True
                    break
                for next_pos in get_neighbors(current):
                    if next_pos in obstacles and next_pos != goal:
                        continue
                    new_cost = cost_so_far[current] + 1
                    if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                        cost_so_far[next_pos] = new_cost
                        priority = new_cost + manhattan(goal, next_pos)
                        heapq.heappush(frontier, (priority, next_pos))
                        came_from[next_pos] = current
            if found:
                # Reconstruct the path.
                current = goal
                path = []
                while current != start:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                if path:
                    next_step = path[0]
                    dx = next_step[0] - start[0]
                    dy = next_step[1] - start[1]
                    return (dx, dy)
            # Fallback: choose any valid direction.
            for d in [(1,0), (-1,0), (0,1), (0,-1)]:
                new_head = (start[0] + d[0], start[1] + d[1])
                if in_bounds(new_head) and new_head not in obstacles:
                    return d
            return self.ai_direction
        else:
            # This branch should not be reached in compute_ai_direction since versus mode is handled in compute_ai_direction_versus.
            return self.ai_direction
        
    def compute_ai_direction(self):
        if self.mode == "classic":
            if self.ai_model == "PPO":
                return self.compute_ai_direction_PPO()
            else:
                return self.compute_ai_direction_astar()
        else:
            # versus-mode already has two methods
            if self.ai_model == "PPO":
                return self.compute_ai_direction_versus_PPO()
            else:
                return self.compute_ai_direction_versus()


    def compute_ai_direction_versus(self):
        # Optimization: Reuse a cached path if it exists and is still valid.
        if hasattr(self, 'ai_path') and self.ai_path:
            next_step = self.ai_path[0]
            obstacles = set(self.player_snake + self.ai_snake)
            if next_step not in obstacles and in_bounds(next_step):
                self.ai_path.pop(0)
                head = self.ai_snake[0]
                return (next_step[0] - head[0], next_step[1] - head[1])
            else:
                self.ai_path = []  # Invalidate cached path if the next step is blocked

        # Compute the size difference.
        size_diff = (len(self.ai_snake) - 1) - (len(self.player_snake) - 1)
        aggressive = size_diff > 3  # AI becomes aggressive when ahead by more than 3 apples

        # Set the goal based on aggression:
        # In aggressive mode, target the player's head.
        # Otherwise, target the apple.
        if aggressive:
            player_head = self.player_snake[0]
            goal = (
            player_head[0] + self.player_direction[0],
            player_head[1] + self.player_direction[1]
            )
        else:
            goal = self.apple

        start = self.ai_snake[0]
        obstacles = set(self.player_snake + self.ai_snake)
        frontier = []
        heapq.heappush(frontier, (0, start))
        came_from = {start: None}
        cost_so_far = {start: 0}

        while frontier:
            current_priority, current = heapq.heappop(frontier)
            if current == goal:
                break
            for next_pos in get_neighbors(current):
                # Allow the goal even if it’s among obstacles.
                if next_pos in obstacles and next_pos != goal:
                    continue
                bonus = 0
                # Standard bonus: if next_pos is adjacent to any part of the player's snake,
                # increase its desirability slightly (only applies in nonaggressive mode here).
                for cell in self.player_snake:
                    if manhattan(next_pos, cell) == 1:
                        bonus = 3
                        break
                step_cost = max(1 - bonus, 0)
                new_cost = cost_so_far[current] + step_cost
                new_priority = new_cost + manhattan(goal, next_pos)
                if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                    cost_so_far[next_pos] = new_cost
                    heapq.heappush(frontier, (new_priority, next_pos))
                    came_from[next_pos] = current

        if goal in came_from:
            # Reconstruct the path from start to goal.
            path = []
            current = goal
            while current != start:
                path.append(current)
                current = came_from[current]
            path.reverse()
            # Cache the computed path.
            self.ai_path = path[:]
            if path:
                next_step = path[0]
                return (next_step[0] - start[0], next_step[1] - start[1])
        # Fallback: choose any valid direction.
        for d in [(1,0), (-1,0), (0,1), (0,-1)]:
            new_head = (start[0] + d[0], start[1] + d[1])
            if in_bounds(new_head) and new_head not in obstacles:
                return d
        return self.ai_direction

    def move(self):
        if self.game_over:
            return
        
        if MOVE_INTERVAL > 0:
            self.process_direction_queue()
        
        if self.mode == "versus":
            # In versus mode, update the AI direction using the versus function.
            if not self.ai_dead:
                self.ai_direction = self.compute_ai_direction()
            # Compute new head positions only for alive snakes.
            new_head_player = None
            if not self.player_dead:
                new_head_player = (self.player_snake[0][0] + self.player_direction[0],
                                   self.player_snake[0][1] + self.player_direction[1])
            new_head_ai = None
            if not self.ai_dead:
                new_head_ai = (self.ai_snake[0][0] + self.ai_direction[0],
                               self.ai_snake[0][1] + self.ai_direction[1])
            # Check collisions.
            if not self.player_dead:
                did_player_die_to_AI = new_head_player in self.ai_snake
                did_player_die_to_player = new_head_player in self.player_snake
                did_player_die_to_wall = not in_bounds(new_head_player)
                if (did_player_die_to_wall or
                    did_player_die_to_player or
                    did_player_die_to_AI):  # collision with enemy snake
                    self.player_dead = True
                    if did_player_die_to_AI:
                        self.player_died_by = "AI"
                    elif did_player_die_to_player:
                        self.player_died_by = "Player"
                    else:
                        self.player_died_by = "Wall" 
            if not self.ai_dead:
                if (not in_bounds(new_head_ai) or
                    new_head_ai in self.ai_snake or
                    new_head_ai in self.player_snake):  # collision with enemy snake
                    self.ai_dead = True
            # End the game if one snake dies.
            if self.player_dead or self.ai_dead:
                self.game_over = True
                self.winner = "AI" if self.player_dead else "Player"
                return
            # Otherwise update positions.
            self.player_snake.insert(0, new_head_player)
            if new_head_player == self.apple:
                self.apple = self._generate_apple(self.player_snake, self.ai_snake)
            else:
                self.player_snake.pop()
            self.ai_snake.insert(0, new_head_ai)
            if new_head_ai == self.apple:
                self.apple = self._generate_apple(self.player_snake, self.ai_snake)
            else:
                self.ai_snake.pop()

        else:
            # Classic mode logic.
            if not self.ai_dead:
                self.ai_direction = self.compute_ai_direction()

            new_head_player = None
            if not self.player_dead:
                new_head_player = (self.player_snake[0][0] + self.player_direction[0],
                                   self.player_snake[0][1] + self.player_direction[1])
            new_head_ai = None
            if not self.ai_dead:
                new_head_ai = (self.ai_snake[0][0] + self.ai_direction[0],
                               self.ai_snake[0][1] + self.ai_direction[1])

            if not self.player_dead and (not in_bounds(new_head_player) or new_head_player in self.player_snake):
                self.player_dead = True
            if not self.ai_dead and (not in_bounds(new_head_ai) or new_head_ai in self.ai_snake):
                self.ai_dead = True

            if self.player_dead and self.ai_dead:
                self.game_over = True
                if (len(self.player_snake)-1) > (len(self.ai_snake)-1):
                    self.winner = "Player"
                elif (len(self.ai_snake)-1) > (len(self.player_snake)-1):
                    self.winner = "AI"
                else:
                    self.winner = "Tie"
                return

            if not self.player_dead:
                self.player_snake.insert(0, new_head_player)
                if new_head_player == self.player_apple:
                    self.place_player_apple()
                else:
                    self.player_snake.pop()
            if not self.ai_dead:
                self.ai_snake.insert(0, new_head_ai)
                if new_head_ai == self.ai_apple:
                    self.place_ai_apple()
                else:
                    self.ai_snake.pop()

            if not self.player_dead and self.ai_dead:
                if (len(self.player_snake) - 1) > (len(self.ai_snake) - 1):
                    self.game_over = True
                    self.winner = "Player"
                    return
            if not self.ai_dead and self.player_dead:
                if (len(self.ai_snake) - 1) > (len(self.player_snake) - 1):
                    self.game_over = True
                    self.winner = "AI"
                    return
# ------------------ End of Game Class ------------------

game = Game()

# ------------------ Load Trained PPO Model ------------------

# Define and load the model if the mode is set for versus.
ppo_model = ActorCritic(GRID_WIDTH, GRID_HEIGHT, 4).to(DEVICE)
ppo_model.load_state_dict(torch.load("ppo_snake_100000.pth", map_location=DEVICE))
ppo_model.eval()
print("Loaded PPO model from ppo_snake_100000.pth")

# ------------------ Web Socket and Game Loop ------------------

def game_loop():
    while True:
        eventlet.sleep(MOVE_INTERVAL)
        game.move()
        # Emit game state to all connected clients after every move.
        state_payload = {}
        if game.mode == "versus":
            state_payload = {
                'player_snake': game.player_snake,
                'ai_snake': game.ai_snake,
                'apple': game.apple,
                'game_over': game.game_over,
                'winner': game.winner,
                'grid_width': game.grid_width,
                'grid_height': game.grid_height,
                'mode': game.mode,
            }
        else:
            state_payload = {
                'player_snake': game.player_snake,
                'ai_snake': game.ai_snake,
                'player_apple': game.player_apple,
                'ai_apple': game.ai_apple,
                'game_over': game.game_over,
                'winner': game.winner,
                'grid_width': game.grid_width,
                'grid_height': game.grid_height,
                'mode': game.mode,
            }
        print('state playload sent')
        socketio.emit('game_state', state_payload)

socketio.start_background_task(game_loop)

@socketio.on('connect')
def on_connect():
    print("Client connected")
    if game.mode == "versus":
        state_payload = {
            'player_snake': game.player_snake,
            'ai_snake': game.ai_snake,
            'apple': game.apple,
            'game_over': game.game_over,
            'winner': game.winner,
            'grid_width': game.grid_width,
            'grid_height': game.grid_height,
            'mode': game.mode,
        }
    else:
        state_payload = {
            'player_snake': game.player_snake,
            'ai_snake': game.ai_snake,
            'player_apple': game.player_apple,
            'ai_apple': game.ai_apple,
            'game_over': game.game_over,
            'winner': game.winner,
            'grid_width': game.grid_width,
            'grid_height': game.grid_height,
            'mode': game.mode,
        }
    emit('game_state', state_payload)




@socketio.on('change_direction')
def on_change_direction(data):
    direction = data.get('direction')
    mapping = {
        'UP': (0, -1),
        'DOWN': (0, 1),
        'LEFT': (-1, 0),
        'RIGHT': (1, 0),
    }
    if direction in mapping:
        game.change_player_direction(mapping[direction])
    emit('change_direction_ack', {'status': 'ok'})

@socketio.on('new_game')
def on_new_game(data):
    mode       = data.get("mode", "classic")
    board_size = int(data.get("board_size", 20))
    ai_model   = data.get("ai_model", "A_STAR")  # or "PPO"
    print(f"Starting new game: mode={mode}, size={board_size}, ai={ai_model}")
    game.reset(mode, board_size, ai_model)
    emit('new_game_ack', {'status': 'ok'})


@socketio.on('restart_game')
def on_restart_game():
    print("game restarted with:" + game.mode)
    game.reset(game.mode, game.grid_height, game.ai_model)
    emit('restart_game_ack', {'status': 'ok'})

if __name__ == '__main__':
    socketio.run(app, debug=True, use_reloader=False)
