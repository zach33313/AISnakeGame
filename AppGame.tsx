import React, { useState } from 'react';
import VersusSnakeGame from './VersusSnakeGame';
import DualSnakeGame from './SnakeGame'; // Classic mode component.
import RestartButton from './Restart';
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

  // defaults
  const [selectedSize, setSelectedSize] = useState<number>(20);
  const [selectedOpponent, setSelectedOpponent] = useState<'A_STAR' | 'PPO'>('A_STAR');

  const handleRestart = () => {
    setGameKey(Date.now());
    console.log('Game restarted!');
  };

  // Generic new-game emitter
  const startNewGame = async (newMode: 'classic' | 'versus', newModel: 'A_STAR' | 'PPO', newSize: number) => {
    setIsLoading(true);
    try {
      console.log('Starting new game:', { newMode, selectedSize, selectedOpponent });
      socket.emit('new_game', {
        mode: newMode,
        board_size: newSize,
        ai_model: newModel
      });
      // force remount
      setNewGameFlag(flag => flag + 1);
    } catch (err) {
      console.error('Error starting new game:', err);
    } finally {
      setIsLoading(false);
    }
  };

  // Mode switch
  const handleModeChange = async (newMode: 'classic' | 'versus') => {
    setMode(newMode);
    await startNewGame(newMode, selectedOpponent, selectedSize);
  };

  // Board-size switch
  const handleSizeChange = async (e: React.ChangeEvent<HTMLSelectElement>) => {
    const size = parseInt(e.target.value, 10) as 6 | 10 | 20;
    setSelectedSize(size);
    await startNewGame(mode, selectedOpponent, size);
  };

  // Opponent switch
  const handleOpponentChange = async (e: React.ChangeEvent<HTMLSelectElement>) => {
    const model = e.target.value as 'A_STAR' | 'PPO';
    setSelectedOpponent(model);
    await startNewGame(mode, model, selectedSize);
  };

  const toggleDropdown = () => setIsOpen(open => !open);

  return (
    <div className="App">
      <div className='sidebar'>
        <h1>Snake Game</h1>
        <div className="dropdown">
          <img
            src={snakeHead}
            alt="Snake head"
            className="snake-head"
            onClick={toggleDropdown}
          />
          <div className={`tongue-container ${isOpen ? 'show' : ''}`}>
            <img src={snakeTongue} alt="Snake tongue" className="snake-tongue" />
            <div className="tongue-options">
              <button
                onClick={() => handleModeChange('classic')}
                className="apple-button"
              >
                <img src={appleImg} alt="Classic" />
                <span>Classic</span>
              </button>
              <button
                onClick={() => handleModeChange('versus')}
                className="apple-button"
              >
                <img src={appleImg} alt="Slither" />
                <span>Slither</span>
              </button>
            </div>
          </div>
        </div>
        <div className="credits">
          <h3>Chloe Lee</h3>
          <h3>Zach Hixson</h3>
          <h3>Conor Abramson-Tieu</h3>
        </div>
      </div>

      <div className="main-content">
        <div className="settings">
          <label>
            <strong>Board Size:</strong>{' '}
            <select value={selectedSize} onChange={handleSizeChange}>
              <option value={6}>6 × 6</option>
              <option value={10}>10 × 10</option>
              <option value={20}>20 × 20</option>
            </select>
          </label>

          <label>
            <strong>Opponent:</strong>{' '}
            <select value={selectedOpponent} onChange={handleOpponentChange}>
              <option value="A_STAR">A★ (A*)</option>
              <option value="PPO">PPO</option>
            </select>
          </label>
        </div>

        <div className="game-container">
          {isLoading && <div className="loader" />}
          {!isLoading && mode === 'versus' && (
            <VersusSnakeGame key={newGameFlag} />
          )}
          {!isLoading && mode === 'classic' && (
            <DualSnakeGame key={newGameFlag} />
          )}
        </div>

        <div className='restart'>
          <RestartButton onRestart={handleRestart} />
        </div>
      </div>
    </div>
  );
};

export default AppGame;
