import React, { useEffect, useRef, useState } from 'react';
import { io, Socket } from 'socket.io-client';
import { socket } from './socket';



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

const CELL_SIZE = 20;      // Each grid cell is 20x20 pixels
const EXTRA_HEIGHT = 40;   // Extra space at bottom for snake length text
const ip_and_port = 'http://127.0.0.1:5000/';
const socketServerUrl = 'http://127.0.0.1:5000/';


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
/*
  const fetchGameState = async () => {
    try {
      const response = await fetch(ip_and_port + 'state');
      const state: VersusGameState = await response.json();
      setGameState(state);
    } catch (error) {
      console.error('Error fetching game state:', error);
    }
  };
*/
  /*const sendDirection = async (direction: string) => {
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
*/
  const sendDirection = (direction: string) => {
    socket?.emit('change_direction', { direction });
  };

  const drawCanvas = (canvas: HTMLCanvasElement, gameState: VersusGameState) => {
    const ctx = canvas.getContext('2d');
    if (ctx) {
      const width = gameState.grid_width * CELL_SIZE;
      const height = gameState.grid_height * CELL_SIZE;
      // Set canvas size: extra height for text at bottom.
      canvas.width = width;
      canvas.height = height + EXTRA_HEIGHT;
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Draw grid outline.
      ctx.strokeStyle = 'black';
      ctx.lineWidth = 2;
      ctx.strokeRect(0, 0, width, height);

      // Draw player snake (green).
      ctx.fillStyle = 'green';
      gameState.player_snake.forEach(cell => {
        ctx.fillRect(cell[0] * CELL_SIZE, cell[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE);
      });

      // Draw AI snake (blue).
      ctx.fillStyle = 'blue';
      gameState.ai_snake.forEach(cell => {
        ctx.fillRect(cell[0] * CELL_SIZE, cell[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE);
      });

      // Draw the shared apple (red).
      ctx.fillStyle = 'red';
      ctx.fillRect(
        gameState.apple[0] * CELL_SIZE,
        gameState.apple[1] * CELL_SIZE,
        CELL_SIZE,
        CELL_SIZE
      );

      // Draw the snake lengths.
      const playerLength = gameState.player_snake.length;
      const aiLength = gameState.ai_snake.length;
      ctx.fillStyle = 'black';
      ctx.font = '20px Arial';
      ctx.textAlign = 'center';
      // We display two lines of text at the bottom.
      ctx.fillText(
        `Player Length: ${playerLength} (${playerLength - 1} apples)`,
        width / 2,
        height + EXTRA_HEIGHT - 20
      );
      ctx.fillText(
        `AI Length: ${aiLength} (${aiLength - 1} apples)`,
        width / 2,
        height + EXTRA_HEIGHT - 0
      );
    }
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
            backgroundColor: 'rgba(255,255,255,0.8)',
          }}
        >
          <h2>Game Over! Winner: {gameState.winner}</h2>
        </div>
      )}
    </div>
  );
};

export default VersusSnakeGame;
