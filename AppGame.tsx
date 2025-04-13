import React, { useState } from 'react';
import VersusSnakeGame from './VersusSnakeGame';
import DualSnakeGame from './SnakeGame'; // Classic mode component.
import RestartButton from './restart';
import { socket } from './socket';  // Import our shared socket connection.
import './AppGame.css';

// Ensure these file paths point to your actual snake images:
import snakeHead from './snakehead-removebg-preview.png';
import snakeTongue from './snaketongue-removebg-preview.png';
import appleImg from './apple-removebg-preview.png';


const AppGame: React.FC = () => {
  const [mode, setMode] = useState<'classic' | 'versus'>('classic');
  const [newGameFlag, setNewGameFlag] = useState(0);
  const [isLoading, setIsLoading] = useState(false);
  const [isOpen, setIsOpen] = useState(false);
  const [gameKey, setGameKey] = useState(Date.now());
  const handleRestart = () => {
    setGameKey(Date.now());
    console.log('Game restarted!');
  };


  // Start a new game by emitting a socket event.
  const startNewGame = async (newMode: string) => {
    setIsLoading(true);
    try {
      console.log("Starting new game in mode:", newMode);
      // Emit 'new_game' with the mode payload.
      socket.emit('new_game', { mode: newMode });
      // Increment flag to force re-mounting of the game component.
      setNewGameFlag(prev => prev + 1);
    } catch (error) {
      console.error('Error starting new game:', error);
    } finally {
      setIsLoading(false);
    }
  };

  // Handle mode change: update mode state and then start a new game.
  const handleModeChange = async (e: String) => {
    const newMode = e as 'classic' | 'versus';
    console.log('Selected mode from event:', newMode);
    setMode(newMode);
    await startNewGame(newMode);
  };

  const toggleDropdown = () => {
    setIsOpen((prev) => !prev);
  };

  return (
    <div className="App">
      <h1>Snake Game</h1>
      <div className="dropdown">
      {/* The enlarged snake head toggles the dropdown */}
      <img
        src={snakeHead}
        alt="Snake head"
        className="snake-head"
        onClick={toggleDropdown}
      />

      {/* Enlarge the tongue and display apple buttons when open */}
      <div className={`tongue-container ${isOpen ? 'show' : ''}`}>
        <img src={snakeTongue} alt="Snake tongue" className="snake-tongue" />
        <div className="tongue-options">
          <button onClick={async () => await handleModeChange("classic")} className="apple-button">
            <img src={appleImg} alt="Apple for Classic" />
            <span>Classic</span>
          </button>
          <button onClick={async () => await handleModeChange("versus")} className="apple-button">
            <img src={appleImg} alt="Apple for Slither" />
            <span>Slither</span>
          </button>
        </div>
      </div>
    </div>
      <div>
        {mode === 'versus' && !isLoading && (
          <VersusSnakeGame key={newGameFlag} />
        )}
        {mode === 'classic' && !isLoading && (
          <DualSnakeGame key={newGameFlag} />
        )}
        {isLoading && (<div className="loader"></div>)}
      </div>
      <div style={{ marginTop: '20px' }}>
        <RestartButton onRestart={handleRestart} />
      </div>
    </div>
  );
};

export default AppGame;
