// src/ModelView.tsx
import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { Line } from 'react-chartjs-2';
import {
  Chart,
  LineElement,
  PointElement,
  CategoryScale,
  LinearScale,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';

// Register required Chart.js components.
Chart.register(LineElement, PointElement, CategoryScale, LinearScale, Title, Tooltip, Legend);

// ------------------ Type Definitions ------------------
interface TrainingServer {
  id: string;
  url: string;
}

interface ProgressPoint {
  episode: number;
  avg_reward: number;
}

interface TrainingStatus {
  current_episode: number;
  avg_reward: number;
  board_size: number | null;
}

interface BestReplay {
  best_reward: number;
  replay: ReplaySnapshot[];
}

interface ReplaySnapshot {
  snake: number[][];
  apple: number[];
}

// ------------------ Hardcoded Training Server Config ------------------
const trainingServers: TrainingServer[] = [
  { id: 'Server 1', url: 'put_ur_url_and_port_here' },
  // Add additional servers as needed.
];

// ------------------ TrainingSession Component ------------------
interface TrainingSessionProps {
  server: TrainingServer;
}

const TrainingSession: React.FC<TrainingSessionProps> = ({ server }) => {
  // Local state for input parameters.
  const [boardSize, setBoardSize] = useState<number>(20);
  const [episodes, setEpisodes] = useState<number>(1000);
  // State for training status from the server.
  const [status, setStatus] = useState<TrainingStatus | null>(null);
  const [progress, setProgress] = useState<ProgressPoint[]>([]);
  const [bestReplay, setBestReplay] = useState<BestReplay | null>(null);
  // isTraining indicates whether a training session is active.
  const [isTraining, setIsTraining] = useState<boolean>(false);
  // paused indicates if training is currently paused.
  const [paused, setPaused] = useState<boolean>(false);

  // Poll the training status every 10 seconds only when training is active and not paused.
  useEffect(() => {
    if (!isTraining || paused) return; // Do not poll if training is stopped or paused.
    const interval = setInterval(() => {
      axios.get(`${server.url}/get_training_status`)
        .then((res) => {
          const data = res.data as TrainingStatus;
          setStatus(data);
          setProgress(prev => [
            ...prev,
            { episode: data.current_episode, avg_reward: data.avg_reward }
          ]);
        })
        .catch((err) =>
          console.error(`Error fetching status from ${server.id}:`, err)
        );
    }, 10000);
    return () => clearInterval(interval);
  }, [server, isTraining, paused]);

  // Start training: calls the backend /start_train route.
  const handleStartTraining = () => {
    axios.post(`${server.url}/start_train`, { size: boardSize, episodes })
      .then((res) => {
        alert(res.data.status);
        setIsTraining(true);
        setPaused(false);
        setProgress([]); // clear old progress
      })
      .catch((err) => {
        console.error(`Error starting training on ${server.id}:`, err);
        alert("Error starting training.");
      });
  };

  const handlePauseTraining = (pause: boolean) => {
    axios.post(`${server.url}/pause_train`, { pause })
      .then((res) => {
        alert(res.data.status);
        setPaused(pause);
      })
      .catch((err) => alert("Error pausing/resuming training."));
  };

  const handleStopTraining = () => {
    axios.post(`${server.url}/stop_train`)
      .then((res) => {
        alert(res.data.status);
        setIsTraining(false);
        setPaused(false);
      })
      .catch((err) => alert("Error stopping training."));
  };

  const handleDownloadModel = () => {
    window.open(`${server.url}/download_model_file`, '_blank');
  };

  const handleFetchBestReplay = () => {
    axios.get(`${server.url}/get_best_replay`)
      .then((res) => setBestReplay(res.data as BestReplay))
      .catch((err) =>
        console.error(`Error fetching best replay from ${server.id}:`, err)
      );
  };

  return (
    <div style={{ border: '1px solid #ccc', margin: '1rem', padding: '1rem' }}>
      <h2>{server.id}</h2>
      {!isTraining ? (
        <div>
          <h3>Start Training</h3>
          <label>
            Board Size:&nbsp;
            <select value={boardSize} onChange={(e) => setBoardSize(parseInt(e.target.value))}>
              <option value={6}>6x6</option>
              <option value={10}>10x10</option>
              <option value={20}>20x20</option>
            </select>
          </label>
          <br />
          <label>
            Episodes:&nbsp;
            <input
              type="number"
              value={episodes}
              onChange={(e) => setEpisodes(parseInt(e.target.value))}
            />
          </label>
          <br />
          <button onClick={handleStartTraining}>Start Training</button>
        </div>
      ) : (
        <div>
          <p>Training in progress...</p>
          {paused && <p style={{ color: 'red' }}>Training is paused.</p>}
          {status && (
            <>
              <p>Current Episode: {status.current_episode}</p>
              <p>Average Reward: {status.avg_reward.toFixed(2)}</p>
            </>
          )}
          <div style={{ maxWidth: '600px' }}>
            <Line
              data={{
                labels: progress.map((p) => p.episode),
                datasets: [
                  {
                    label: 'Average Reward',
                    data: progress.map((p) => p.avg_reward),
                    fill: false,
                    borderColor: 'rgba(75,192,192,1)',
                  },
                ],
              }}
              options={{
                scales: {
                  x: { title: { display: true, text: 'Episode' } },
                  y: { title: { display: true, text: 'Average Reward' } },
                },
              }}
            />
          </div>
          <div style={{ marginTop: '1rem' }}>
            <button onClick={() => handlePauseTraining(true)}>Pause Training</button>
            <button onClick={() => handlePauseTraining(false)} style={{ marginLeft: '1rem' }}>
              Resume Training
            </button>
            <button onClick={handleStopTraining} style={{ marginLeft: '1rem' }}>
              Stop Training
            </button>
            <button onClick={handleDownloadModel} style={{ marginLeft: '1rem' }}>
              Download Model
            </button>
            <button onClick={handleFetchBestReplay} style={{ marginLeft: '1rem' }}>
              Show Best Replay
            </button>
          </div>
          {bestReplay && (
            <div style={{ marginTop: '1rem' }}>
              <h3>Best Replay (Reward: {bestReplay.best_reward})</h3>
              <ReplayCanvas replay={bestReplay.replay} boardSize={status?.board_size ?? 20} />
            </div>
          )}
        </div>
      )}
    </div>
  );
};

// ------------------ ReplayCanvas Component ------------------
interface ReplayCanvasProps {
  replay: ReplaySnapshot[];
  boardSize: number;
  cellSize?: number;
}

const ReplayCanvas: React.FC<ReplayCanvasProps> = ({ replay, boardSize, cellSize = 20 }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!replay || replay.length === 0) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    canvas.width = boardSize * cellSize;
    canvas.height = boardSize * cellSize;
    let frame = 0;
    const interval = setInterval(() => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      // Draw grid
      ctx.strokeStyle = '#ddd';
      for (let i = 0; i < boardSize; i++) {
        for (let j = 0; j < boardSize; j++) {
          ctx.strokeRect(j * cellSize, i * cellSize, cellSize, cellSize);
        }
      }
      const snapshot = replay[frame];
      // Draw snake in green
      ctx.fillStyle = 'green';
      snapshot.snake.forEach(([x, y]) => {
        ctx.fillRect(y * cellSize, x * cellSize, cellSize, cellSize);
      });
      // Draw apple in red
      ctx.fillStyle = 'red';
      const [ax, ay] = snapshot.apple;
      ctx.fillRect(ay * cellSize, ax * cellSize, cellSize, cellSize);
      frame++;
      if (frame >= replay.length) {
        clearInterval(interval);
      }
    }, 200); // Frame delay in ms.
    return () => clearInterval(interval);
  }, [replay, boardSize, cellSize]);

  return <canvas ref={canvasRef} style={{ border: '1px solid #ccc' }} />;
};

// ------------------ ModelView Component ------------------
const ModelView: React.FC = () => {
  return (
    <div style={{ padding: '1rem' }}>
      <h1>Model Training Dashboard</h1>
      {trainingServers.map((server) => (
        <TrainingSession key={server.id} server={server} />
      ))}
    </div>
  );
};

export default ModelView;
