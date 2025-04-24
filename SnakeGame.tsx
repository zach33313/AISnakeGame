// DualSnakeGame.tsx
import React, { useEffect, useRef, useState } from 'react';
import { socket } from './socket';
import appleImg from './apple-removebg-preview.png';

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
// apple image import
const appleImage = new Image();
appleImage.src = appleImg;

export const drawCanvas = (
  canvas: HTMLCanvasElement,
  snake: [number, number][],
  apple: [number, number],
  label: string,
  grid_width: number,
  grid_height: number
) => {
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  const drawEyes = (x: number, y: number, outlineColor: string) => {
    const centerX = x * CELL_SIZE + CELL_SIZE / 2;
    const centerY = y * CELL_SIZE + CELL_SIZE / 2;
    const eyeOffset = 7;
    const eyeRadius = 6;

    ctx.fillStyle = 'white';
    ctx.strokeStyle = outlineColor;
    ctx.lineWidth = 1;

    // left eye
    ctx.beginPath();
    ctx.arc(centerX - eyeOffset, centerY - eyeOffset, eyeRadius, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    // right eye
    ctx.beginPath();
    ctx.arc(centerX + eyeOffset, centerY - eyeOffset, eyeRadius, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = 'black';
    ctx.beginPath();
    ctx.arc(centerX - eyeOffset, centerY - eyeOffset, 2, 0, Math.PI * 2);
    ctx.arc(centerX + eyeOffset, centerY - eyeOffset, 2, 0, Math.PI * 2);
    ctx.fill();
  };

  const width = grid_width * CELL_SIZE;
  const height = grid_height * CELL_SIZE;
  canvas.width = width;
  canvas.height = height + EXTRA_HEIGHT;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // border
  ctx.strokeStyle = 'black';
  ctx.lineWidth = 2;
  ctx.strokeRect(0, 0, width, height);

  const drawSnake = (
    snake: [number, number][],
    color: string,
    headColor: string
  ) => {
    snake.forEach(([x, y], index) => {
      const px = x * CELL_SIZE;
      const py = y * CELL_SIZE;
      ctx.fillStyle = index === 0 ? headColor : color;
      ctx.beginPath();
      ctx.roundRect(px, py, CELL_SIZE, CELL_SIZE, 6);
      ctx.fill();
    });
  };

  const isPlayer = label === "Player";
  const bodyColor = isPlayer ? '#77dd77' : '#89CFF0';
  const headColor = isPlayer ? '#32a852' : '#1f75fe';
  const eyeOutline = isPlayer ? '#1c5c32' : '#1a4ba0';

  drawSnake(snake, bodyColor, headColor);
  drawEyes(snake[0][0], snake[0][1], eyeOutline);

  // apple rendering
  if (appleImage.complete) {
    ctx.drawImage(
      appleImage,
      apple[0] * CELL_SIZE,
      apple[1] * CELL_SIZE,
      CELL_SIZE,
      CELL_SIZE
    );
  }

  // score text
  ctx.fillStyle = 'black';
  ctx.font = '20px Arial';
  ctx.textAlign = 'center';
  ctx.fillText(`${label} Length: ${snake.length} (${snake.length - 1} apples)`, width / 2, height + 20);
};

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

  useEffect(() => {
    if (gameState) {
      if (playerCanvasRef.current) {
        drawCanvas(
          playerCanvasRef.current,
          gameState.player_snake,
          gameState.player_apple,
          "Player",
          gameState.grid_width,
          gameState.grid_height
        );
      }
      if (aiCanvasRef.current) {
        drawCanvas(
          aiCanvasRef.current,
          gameState.ai_snake,
          gameState.ai_apple,
          "AI",
          gameState.grid_width,
          gameState.grid_height
        );
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
