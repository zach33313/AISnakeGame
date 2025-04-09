// DualSnakeGame.tsx
import React, { useEffect, useRef, useState } from 'react';
import { socket } from './socket';

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

const CELL_SIZE = 20;      // Each grid cell is 20x20 pixels.
const EXTRA_HEIGHT = 30;   // Extra space at bottom for text.

const DualSnakeGame: React.FC = () => {
  const playerCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const aiCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const [gameState, setGameState] = useState<GameState | null>(null);

  // Listen for game state updates from the server.
  useEffect(() => {
    socket.on('game_state', (state: GameState) => {
      console.log('recieved game state')
      setGameState(state);
    });
    return () => {
      socket.off('game_state');
    };
  }, []);

  // Listen for arrow key events and send direction events over the socket.
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      let direction = '';
      if (event.key === 'ArrowUp') direction = 'UP';
      else if (event.key === 'ArrowDown') direction = 'DOWN';
      else if (event.key === 'ArrowLeft') direction = 'LEFT';
      else if (event.key === 'ArrowRight') direction = 'RIGHT';
      if (direction) {
        socket.emit('change_direction', { direction });
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

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

      // Draw snake.
      ctx.fillStyle = label === "Player" ? 'green' : 'blue';
      snake.forEach(cell => {
        ctx.fillRect(cell[0] * CELL_SIZE, cell[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE);
      });

      // Draw apple.
      ctx.fillStyle = 'red';
      ctx.fillRect(apple[0] * CELL_SIZE, apple[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE);

      // Draw apple count text.
      const appleCount = snake.length - 1;
      ctx.fillStyle = 'black';
      ctx.font = '20px Arial';
      ctx.textAlign = 'center';
      ctx.fillText(`${label} Apple Count: ${appleCount}`, width / 2, height + EXTRA_HEIGHT - 10);
    }
  };

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
