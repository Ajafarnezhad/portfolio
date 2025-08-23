import React, { useState } from 'react';
import Dropzone from './components/Dropzone';
import Result from './components/Result';

function App() {
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState(null);

  const handlePrediction = (result) => {
    setPrediction(result);
    setError(null);
  };

  const handleError = (error) => {
    setError(error);
    setPrediction(null);
  };

  return (
    <div className="min-h-screen bg-gray-100 flex flex-col">
      <header className="bg-indigo-600 text-white py-4">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <h1 className="text-2xl font-bold">Face Mask Detection</h1>
        </div>
      </header>
      <main className="flex-grow max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <Dropzone onPrediction={handlePrediction} onError={handleError} />
        {prediction && <Result prediction={prediction} />}
        {error && (
          <div className="mt-4 bg-red-100 text-red-700 p-4 rounded-lg">
            {error}
          </div>
        )}
      </main>
      <footer className="bg-gray-800 text-white py-4">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <p>&copy; 2025 Face Mask Detection. All rights reserved.</p>
        </div>
      </footer>
    </div>
  );
}

export default App;