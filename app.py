from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
import threading, time, random, heapq

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

GRID_WIDTH = 20
GRID_HEIGHT = 20
MOVE_INTERVAL = 0.2  # seconds between moves

def in_bounds(pos):
    x, y = pos
    return 0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT

def get_neighbors(pos):
    x, y = pos
    neighbors = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
    return [n for n in neighbors if in_bounds(n)]

def manhattan(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

class Game:
    def __init__(self):
        self.grid_width = GRID_WIDTH
        self.grid_height = GRID_HEIGHT
        self.reset()

    def reset(self):
        # Initialize player and AI snakes at different starting positions.
        self.player_snake = [(self.grid_width // 2, self.grid_height // 2)]
        self.ai_snake = [(self.grid_width // 2, self.grid_height // 2 - 3)]
        self.player_direction = (1, 0)  # player starts moving right
        self.ai_direction = (1, 0)      # initial AI direction (will be updated)
        self.place_player_apple()
        self.place_ai_apple()
        self.game_over = False
        self.winner = None

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
        # Prevent the player from reversing direction directly.
        if (self.player_direction[0] * -1, self.player_direction[1] * -1) == new_direction:
            return
        self.player_direction = new_direction

    def compute_ai_direction(self):
        # Use A* search to compute a path from the AI snake's head to its apple.
        start = self.ai_snake[0]
        goal = self.ai_apple
        obstacles = set(self.player_snake + self.ai_snake)
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

    def move(self):
        if self.game_over:
            return

        # Compute AI's next move.
        self.ai_direction = self.compute_ai_direction()

        # Calculate new head positions.
        new_head_player = (self.player_snake[0][0] + self.player_direction[0],
                             self.player_snake[0][1] + self.player_direction[1])
        new_head_ai = (self.ai_snake[0][0] + self.ai_direction[0],
                       self.ai_snake[0][1] + self.ai_direction[1])

        # Check collisions for player.
        player_collision = (
            not in_bounds(new_head_player) or
            new_head_player in self.player_snake 
            #or
            #new_head_player in self.ai_snake
        )
        # Check collisions for AI.
        ai_collision = (
            not in_bounds(new_head_ai) or
            new_head_ai in self.ai_snake 
            #or
            #new_head_ai in self.player_snake
        )
        if player_collision or ai_collision:
            self.game_over = True
            if len(self.player_snake) > len(self.ai_snake):
                self.winner = "Player"
                print(f'ai collided at {new_head_ai}')
            elif len(self.ai_snake) > len(self.player_snake):
                self.winner = "AI"
            else:
                print(f'ai collided at {new_head_ai}')
                self.winner = "Tie"
            return

        # Update snake positions.
        self.player_snake.insert(0, new_head_player)
        self.ai_snake.insert(0, new_head_ai)

        # Check if player's snake eats its apple.
        if new_head_player == self.player_apple:
            self.place_player_apple()
        else:
            self.player_snake.pop()

        # Check if AI's snake eats its apple.
        if new_head_ai == self.ai_apple:
            self.place_ai_apple()
        else:
            self.ai_snake.pop()

game = Game()

def game_loop():
    while True:
        time.sleep(MOVE_INTERVAL)
        game.move()

# Start the game loop in a background thread.
threading.Thread(target=game_loop, daemon=True).start()

@app.route('/state', methods=['GET'])
@cross_origin()
def state():
    return jsonify({
        'player_snake': game.player_snake,
        'ai_snake': game.ai_snake,
        'player_apple': game.player_apple,
        'ai_apple': game.ai_apple,
        'game_over': game.game_over,
        'winner': game.winner,
        'grid_width': game.grid_width,
        'grid_height': game.grid_height,
    })

@app.route('/change_direction', methods=['POST'])
@cross_origin()
def change_direction():
    data = request.get_json()
    direction = data.get('direction')
    mapping = {
        'UP': (0, -1),
        'DOWN': (0, 1),
        'LEFT': (-1, 0),
        'RIGHT': (1, 0),
    }
    if direction in mapping:
        game.change_player_direction(mapping[direction])
    return jsonify({'status': 'ok'})

@app.route('/new_game', methods=['GET'])
@cross_origin()
def new_game():
    game.reset()
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(debug=True)
