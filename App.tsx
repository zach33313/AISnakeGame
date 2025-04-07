import React from 'react';
import DualSnakeGame from './SnakeGame';
import RestartButton from './restart';
import './App.css';

const App: React.FC = () => {
  return (
    <div className="App">
      <h1>Snake Game</h1>
      <DualSnakeGame />
    <div><RestartButton/></div>
    </div>
  );
};

export default App;
