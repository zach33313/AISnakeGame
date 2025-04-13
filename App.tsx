// App.tsx
import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import ModelView from './ModelView';
import AppGame from './AppGame';

const App: React.FC = () => {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<AppGame/>}></Route>
        <Route path="/model_view" element={<ModelView />} />
      </Routes>
    </Router>
  );
};

export default App;
