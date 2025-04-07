import React, { useEffect, useRef, useState } from 'react';

type GameState = {
  player_snake: [number, number][],
  ai_snake: [number, number][],
  player_apple: [number, number],
  ai_apple: [number, number],
  game_over: boolean,
  winner: string | null,
  grid_width: number,
  grid_height: number,
};

const CELL_SIZE = 20;      // Each grid cell is 20x20 pixels
const EXTRA_HEIGHT = 30;   // Extra space at bottom for text
const ip_and_port = 'put_urs_here_dont_forget_the_slash->/';

const DualSnakeGame: React.FC = () => {
  const playerCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const aiCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const [gameState, setGameState] = useState<GameState | null>(null);

  // Poll the game state from the backend.
  const fetchGameState = async () => {
    try {
      const response = await fetch(ip_and_port + 'state');
      const state: GameState = await response.json();
      setGameState(state);
    } catch (error) {
      console.error('Error fetching game state:', error);
    }
  };

  // Send player direction change.
  const sendDirection = async (direction: string) => {
    try {
      await fetch(ip_and_port + 'change_direction', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ direction }),
      });
    } catch (error) {
      console.error('Error sending direction:', error);
    }
  };

  // Draw a canvas for the given snake, apple, and label.
  const drawCanvas = (
    canvas: HTMLCanvasElement,
    snake: [number, number][],
    apple: [number, number],
    label: string
  ) => {
    const ctx = canvas.getContext('2d');
    if (ctx && gameState) {
      const width = gameState.grid_width * CELL_SIZE;
      const height = gameState.grid_height * CELL_SIZE;
      canvas.width = width;
      canvas.height = height + EXTRA_HEIGHT;
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Draw grid outline.
      ctx.strokeStyle = 'black';
      ctx.lineWidth = 2;
      ctx.strokeRect(0, 0, width, height);

      // Draw the snake.
      ctx.fillStyle = label === "Player" ? 'green' : 'blue';
      snake.forEach(cell => {
        ctx.fillRect(cell[0] * CELL_SIZE, cell[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE);
      });

      // Draw the apple specific to this snake.
      ctx.fillStyle = 'red';
      ctx.fillRect(apple[0] * CELL_SIZE, apple[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE);

      // Draw apple count text (snake length - 1, since initial length is 1).
      const appleCount = snake.length - 1;
      ctx.fillStyle = 'black';
      ctx.font = '20px Arial';
      ctx.textAlign = 'center';
      ctx.fillText(`${label} Apple Count: ${appleCount}`, width / 2, height + EXTRA_HEIGHT - 10);
    }
  };

  // Set up key listener and polling.
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      let direction = '';
      if (event.key === 'ArrowUp') direction = 'UP';
      else if (event.key === 'ArrowDown') direction = 'DOWN';
      else if (event.key === 'ArrowLeft') direction = 'LEFT';
      else if (event.key === 'ArrowRight') direction = 'RIGHT';
      if (direction) {
        sendDirection(direction);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    const interval = setInterval(fetchGameState, 100);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      clearInterval(interval);
    };
  }, []);

  // Redraw canvases on game state update.
  useEffect(() => {
    if (gameState) {
      if (playerCanvasRef.current) {
        drawCanvas(playerCanvasRef.current, gameState.player_snake, gameState.player_apple, "Player");
      }
      if (aiCanvasRef.current) {
        drawCanvas(aiCanvasRef.current, gameState.ai_snake, gameState.ai_apple, "AI");
      }
    }
  }, [gameState]);

  return (
    <div style={{ display: 'flex', justifyContent: 'center', gap: '20px', position: 'relative' }}>
      <canvas ref={playerCanvasRef} />
      <canvas ref={aiCanvasRef} />
      {gameState && gameState.game_over && (
        <div style={{ 
          position: 'absolute', 
          top: '10px', 
          width: '100%', 
          textAlign: 'center',
          backgroundColor: 'rgba(255,255,255,0.8)' 
        }}>
          <h2>Game Over! Winner: {gameState.winner}</h2>
        </div>
      )}
    </div>
  );
};

export default DualSnakeGame;
