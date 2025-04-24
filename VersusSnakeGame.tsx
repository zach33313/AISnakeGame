import React, { useEffect, useRef, useState } from 'react';
import { io, Socket } from 'socket.io-client';
import { socket } from './socket';
import appleImg from './apple-removebg-preview.png';


type VersusGameState = {
  player_snake: [number, number][],
  ai_snake: [number, number][],
  apple: [number, number],
  game_over: boolean,
  winner: string | null,
  grid_width: number,
  grid_height: number,
  mode: string,
};

// Fixed canvas dimensions
const CANVAS_WIDTH = 400;   // Fixed width in pixels
const CANVAS_HEIGHT = 400;  // Fixed height in pixels
const EXTRA_HEIGHT = 50;    // Extra space at bottom for snake length text
// apple image import
const appleImage = new Image();
appleImage.src = appleImg;


const VersusSnakeGame: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [gameState, setGameState] = useState<VersusGameState | null>(null);



  // Initialize socket connection on component mount.
  useEffect(() => {
    // Listen for game state updates from the server.
    socket.on('game_state', (state: VersusGameState) => {
      setGameState(state);
      console.log('recieved game state')
    });
    // Clean up the listener on unmount.
    return () => {
      socket.off('game_state');
    };
  }, []);

  const sendDirection = (direction: string) => {
    socket?.emit('change_direction', { direction });
  };

  const drawCanvas = (canvas: HTMLCanvasElement, gameState: VersusGameState) => {
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Calculate cell size dynamically based on grid dimensions
    const CELL_SIZE = Math.min(
      CANVAS_WIDTH / gameState.grid_width,
      CANVAS_HEIGHT / gameState.grid_height
    );

    const drawEyes = (x: number, y: number, outlineColor: string) => {
      const centerX = x * CELL_SIZE + CELL_SIZE / 2;
      const centerY = y * CELL_SIZE + CELL_SIZE / 2;
      const eyeOffset = CELL_SIZE * 0.35; // Scale eye position with cell size
      const eyeRadius = CELL_SIZE * 0.3;  // Scale eye size with cell size

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
      ctx.arc(centerX - eyeOffset, centerY - eyeOffset, CELL_SIZE * 0.1, 0, Math.PI * 2);
      ctx.arc(centerX + eyeOffset, centerY - eyeOffset, CELL_SIZE * 0.1, 0, Math.PI * 2);
      ctx.fill();
    };

    // Set canvas to fixed dimensions
    canvas.width = CANVAS_WIDTH;
    canvas.height = CANVAS_HEIGHT + EXTRA_HEIGHT;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // checkerboard background
    for (let row = 0; row < gameState.grid_height; row++) {
      for (let col = 0; col < gameState.grid_width; col++) {
        ctx.fillStyle = (row + col) % 2 === 0 ? '#ffffff' : '#e5e5e5';
        ctx.fillRect(col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE);
      }
    }

    // Grid border
    ctx.strokeStyle = 'black';
    ctx.lineWidth = 2;
    ctx.strokeRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);

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
        ctx.roundRect(px, py, CELL_SIZE, CELL_SIZE, CELL_SIZE * 0.3);
        ctx.fill();
      });
    };

    // Draw snakes diff colors for player and ai
    drawSnake(gameState.player_snake, '#77dd77', '#32a852'); // Green snake for player
    drawSnake(gameState.ai_snake, '#89CFF0', '#1f75fe');      // Blue snake for ai

    // draw eyes (w/ diff colored outlines)
    const [px, py] = gameState.player_snake[0];
    const [aiX, aiY] = gameState.ai_snake[0];
    drawEyes(px, py, '#1c5c32');
    drawEyes(aiX, aiY, '#1a4ba0');

    // apple image rendering
    const [ax, ay] = gameState.apple;
    if (appleImage.complete) {
      ctx.drawImage(
        appleImage,
        ax * CELL_SIZE,
        ay * CELL_SIZE,
        CELL_SIZE,
        CELL_SIZE
      );
    }

    // Text stats (score)
    ctx.fillStyle = 'black';
    ctx.font = '16px Arial';
    ctx.textAlign = 'center';
    const playerLength = gameState.player_snake.length;
    const aiLength = gameState.ai_snake.length;
    ctx.fillText(`Player Length: ${playerLength} (${playerLength - 1} apples)`, CANVAS_WIDTH / 2, CANVAS_HEIGHT + 20);
    ctx.fillText(`AI Length: ${aiLength} (${aiLength - 1} apples)`, CANVAS_WIDTH / 2, CANVAS_HEIGHT + 40);
  };


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
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [socket]);

  useEffect(() => {
    if (gameState && canvasRef.current) {
      drawCanvas(canvasRef.current, gameState);
    }
  }, [gameState]);

  return (
    <div style={{ display: 'flex', justifyContent: 'center', position: 'relative' }}>
      <canvas ref={canvasRef} />
      {gameState && gameState.game_over && (
        <div
          style={{
            position: 'absolute',
            top: '10px',
            width: '100%',
            textAlign: 'center',
            backgroundColor: 'rgba(255,255,255,0.8)'
          }}
        >
          <h2>Game Over! 🎉 Winner: {gameState.winner}</h2>
        </div>
      )}
    </div>
  );
};

export default VersusSnakeGame; 
