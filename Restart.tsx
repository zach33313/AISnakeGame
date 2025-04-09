// RestartButton.tsx
import React from 'react';
import { socket } from './socket';
import './restart.css'

interface RestartButtonProps {
  onRestart: () => void;
}

const RestartButton: React.FC<RestartButtonProps> = ({ onRestart }) => {
  const onClickRestart = () => {
    console.log('hello')
    // Emit a socket event for restarting the game.
    socket.emit('restart_game');
  };

  return (
    <a
      href="#"
      className="btn"
      onClick={(e) => {
        e.preventDefault();
        onRestart();
        onClickRestart();
      }}
    >
      Restart
      <svg className="snake-svg" viewBox="0 0 300 70" preserveAspectRatio="none">
        {/* The snake path animates around the button border.
            Its stroke-dashoffset animates from 732 to 0 over 12 seconds, leaving a complete green outline */}
        <rect
          className="snake-path"
          x="1"
          y="1"
          width="298"
          height="68"
          rx="8"
          ry="8"
          fill="none"
          stroke="#39FF14"
          strokeWidth="6"
          strokeDasharray="732"
          strokeDashoffset="732"
        />
        {/* The fixed red apple at the top-left which fades out when the snake reaches it */}
        <circle className="apple" cx="1" cy="1" r="6" fill="red" />
      </svg>
    </a>
  );
};

export default RestartButton;
