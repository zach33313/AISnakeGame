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

// Register Chart.js components
Chart.register(LineElement, PointElement, CategoryScale, LinearScale, Title, Tooltip, Legend);

// ------------- Type Definitions -------------
interface TrainingServer { id: string; url: string }
interface ProgressPoint { episode: number; avg_reward: number }
interface TrainingStatus { current_episode: number; avg_reward: number; board_size: number }
interface EvalStep { snake: [number, number][]; apple: [number, number]; game_over: boolean }
interface BestReplay {
    best_reward: number;
    replay: ReplaySnapshot[];
  }
  
  interface ReplaySnapshot {
    snake: number[][];
    apple: number[];
  }
  

// ------------- Config -------------
const trainingServers: TrainingServer[] = [
  { id: 'Server 1', url: 'ur_ip_and_port_here' }
];
const POLL_INTERVAL_MS = 10000     // 10s for training status
const EVAL_STEP_INTERVAL = 200     // ms between eval frames
const CELL_SIZE = 20
const EXTRA_HEIGHT = 30

// ------------- Canvas Drawing Helper -------------
const drawCanvas = (
  canvas: HTMLCanvasElement,
  snake: [number, number][],
  apple: [number, number],
  gridW: number,
  gridH: number,
) => {
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  const width = gridW * CELL_SIZE;
  const height = gridH * CELL_SIZE;
  canvas.width = width;
  canvas.height = height + EXTRA_HEIGHT;

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Grid outline
  ctx.strokeStyle = 'black';
  ctx.lineWidth = 2;
  ctx.strokeRect(0, 0, width, height);

  // Snake
  ctx.fillStyle = 'blue';
  snake.forEach(([x, y]) => {
    ctx.fillRect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE);
  });

  // Apple
  ctx.fillStyle = 'red';
  ctx.fillRect(apple[0] * CELL_SIZE, apple[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE);

  // Apple count text
  const appleCount = snake.length - 1;
  ctx.fillStyle = 'black';
  ctx.font = '16px Arial';
  ctx.textAlign = 'center';
  ctx.fillText(`Apple Count: ${appleCount}`, width / 2, height + EXTRA_HEIGHT - 8);
};

// ------------- EvalSnakeGame Component -------------
interface EvalSnakeGameProps {
  serverUrl: string;
  boardSize: number;
  episode: number;
  onClose: () => void;
}

const EvalSnakeGame: React.FC<EvalSnakeGameProps> = ({
  serverUrl, boardSize, episode, onClose
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [gameOver, setGameOver] = useState(false);

  useEffect(() => {
    axios.post(`${serverUrl}/start_evaluate`, { episode, board_size: boardSize })
      .catch(console.error);

    const interval = setInterval(() => {
      axios.get<EvalStep>(`${serverUrl}/eval_step`)
        .then(res => {
          const { snake, apple, game_over } = res.data;
          if (canvasRef.current) {
            drawCanvas(canvasRef.current, snake, apple, boardSize, boardSize);
          }
          if (game_over) {
            setGameOver(true);
            clearInterval(interval);
          }
        })
        .catch(console.error);
    }, EVAL_STEP_INTERVAL);

    return () => {
      clearInterval(interval);
      axios.post(`${serverUrl}/stop_evaluate`).catch(console.error);
    };
  }, [serverUrl, boardSize, episode]);

  return (
    <div className="modal">
      <div className="modal-content">
        <h3>Evaluation: Episode {episode}</h3>
        <canvas ref={canvasRef} style={{ border: '1px solid #ccc' }} />
        {gameOver && (
          <button onClick={onClose} style={{ marginTop: '10px' }}>
            Close Evaluation
          </button>
        )}
      </div>
    </div>
  );
};

// ------------- TrainingSession Component -------------
const TrainingSession: React.FC<{ server: TrainingServer }> = ({ server }) => {
  const [boardSize, setBoardSize] = useState(20);
  const [episodes, setEpisodes] = useState(1000);
  const [status, setStatus] = useState<TrainingStatus | null>(null);
  const [progress, setProgress] = useState<ProgressPoint[]>([]);
  const [specialPoints, setSpecialPoints] = useState<number[]>([]);
  const [isTraining, setIsTraining] = useState(false);
  const [paused, setPaused] = useState(false);
  const [evalEp, setEvalEp] = useState<number | null>(null);
  const [bestReplay, setBestReplay] = useState<BestReplay | null>(null);


  // Poll training status
  useEffect(() => {
    if (!isTraining || paused) return;
    const iv = setInterval(() => {
      axios.get<TrainingStatus>(`${server.url}/get_training_status`)
        .then(res => {
          setStatus(res.data);
          setProgress(p => [
            ...p,
            { episode: res.data.current_episode, avg_reward: res.data.avg_reward }
          ]);
        })
        .catch(console.error);

      axios.get<{ special_points: number[] }>(`${server.url}/get_special_points`)
        .then(res => setSpecialPoints(res.data.special_points))
        .catch(console.error);
    }, POLL_INTERVAL_MS);

    return () => clearInterval(iv);
  }, [server.url, isTraining, paused]);

  const startTraining = () => {
    axios.post(`${server.url}/start_train`, { size: boardSize, episodes })
      .then(res => {
        alert(res.data.status);
        setIsTraining(true);
        setPaused(false);
        setProgress([]);
      })
      .catch(err => {
        console.error(err);
        alert("Error starting training");
      });
  };

  const togglePause = (p: boolean) => {
    axios.post(`${server.url}/pause_train`, { pause: p })
      .then(res => {
        alert(res.data.status);
        setPaused(p);
      })
      .catch(() => alert("Error toggling pause"));
  };

  const stopTraining = () => {
    axios.post(`${server.url}/stop_train`)
      .then(res => {
        alert(res.data.status);
        setIsTraining(false);
        setPaused(false);
      })
      .catch(() => alert("Error stopping training"));
  };

  const handleFetchBestReplay = () => {
    axios.get(`${server.url}/get_best_replay`)
      .then((res) => setBestReplay(res.data as BestReplay))
      .catch((err) =>
        console.error(`Error fetching best replay from ${server.id}:`, err)
      );
  };


  const onChartClick = (_: any, elements: any[]) => {
    if (!elements.length) return;
    const idx = elements[0].index;
    const ep = progress[idx].episode;
    if (specialPoints.includes(ep)) {
      setEvalEp(ep);
    }
  };

  return (
    <div style={{ border: '1px solid #ccc', padding: '1rem', margin: '1rem' }}>
      <h2>{server.id}</h2>

      {!isTraining ? (
        <>
          <h3>Start Training</h3>
          <label>
            Board Size:&nbsp;
            <select value={boardSize} onChange={e => setBoardSize(+e.target.value)}>
              <option value={6}>6×6</option>
              <option value={10}>10×10</option>
              <option value={20}>20×20</option>
            </select>
          </label>
          <br/>
          <label>
            Episodes:&nbsp;
            <input
              type="number"
              value={episodes}
              onChange={e => setEpisodes(+e.target.value)}
            />
          </label>
          <br/>
          <button onClick={startTraining}>Start Training</button>
        </>
      ) : (
        <>
          <p>Training {paused ? "(paused)" : "in progress"}…</p>
          {status && (
            <>
              <p>Episode: {status.current_episode}</p>
              <p>Avg Reward: {status.avg_reward.toFixed(2)}</p>
            </>
          )}

          <div style={{ maxWidth: 600 }}>
            <Line
              data={{
                labels: progress.map(p => p.episode),
                datasets: [
                  {
                    label: 'Avg Reward',
                    data: progress.map(p => p.avg_reward),
                    fill: false,
                    borderColor: 'teal',
                  },
                  {
                    label: 'Special',
                    data: progress
                      .filter(p => specialPoints.includes(p.episode))
                      .map(p => p.avg_reward),
                    pointBackgroundColor: 'gold',
                    showLine: false,
                    pointRadius: 6,
                  }
                ]
              }}
              options={{
                onClick: onChartClick,
                scales: {
                  x: { title: { display: true, text: 'Episode' } },
                  y: { title: { display: true, text: 'Avg Reward' } },
                },
              }}
            />
          </div>

          <div style={{ marginTop: '1rem' }}>
            <button onClick={() => togglePause(true)}>Pause</button>
            <button onClick={() => togglePause(false)} style={{ marginLeft: 8 }}>Resume</button>
            <button onClick={stopTraining} style={{ marginLeft: 8 }}>Stop</button>
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
        </>
      )}

      {evalEp !== null && status && (
        <EvalSnakeGame
          serverUrl={server.url}
          boardSize={status.board_size!}
          episode={evalEp}
          onClose={() => setEvalEp(null)}
        />
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
        // Draw grid lines
        ctx.strokeStyle = '#ddd';
        for (let r = 0; r < boardSize; r++) {
          for (let c = 0; c < boardSize; c++) {
            ctx.strokeRect(c * cellSize, r * cellSize, cellSize, cellSize);
          }
        }
        const snapshot = replay[frame];
        // Draw snake in green
        ctx.fillStyle = 'green';
        snapshot.snake.forEach(([r, c]) => {
          ctx.fillRect(c * cellSize, r * cellSize, cellSize, cellSize);
        });
        // Draw apple in red
        ctx.fillStyle = 'red';
        const [ar, ac] = snapshot.apple;
        ctx.fillRect(ac * cellSize, ar * cellSize, cellSize, cellSize);
  
        frame++;
        if (frame >= replay.length) {
          clearInterval(interval);
        }
      }, 200);
  
      return () => clearInterval(interval);
    }, [replay, boardSize, cellSize]);
  
    return <canvas ref={canvasRef} style={{ border: '1px solid #ccc' }} />;
  };

// ------------- ModelView Root -------------
export const ModelView: React.FC = () => (
  <div style={{ padding: '1rem' }}>
    <h1>Model Training Dashboard</h1>
    {trainingServers.map(s => (
      <TrainingSession key={s.id} server={s} />
    ))}
  </div>
);

export default ModelView;
