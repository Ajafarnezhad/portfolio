import React from 'react';

function Result({ prediction }) {
  return (
    <div className="mt-6 bg-white shadow-lg rounded-lg p-6 text-center animate-fade-in">
      <h2 className="text-xl font-semibold mb-4">Prediction Result</h2>
      <p className="text-lg mb-4">
        The face is classified as: <span className="font-bold text-indigo-600">{prediction}</span>
      </p>
      <p className="text-gray-600">Thank you for using our service!</p>
    </div>
  );
}

export default Result;