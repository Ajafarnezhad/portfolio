import React from 'react';

function Result({ predictions }) {
  return (
    <div className="mt-6 bg-white shadow-lg rounded-lg p-6 text-center animate-fade-in">
      <h2 className="text-xl font-semibold mb-4">Prediction Results</h2>
      <p className="text-lg mb-4">Predicted ventilator pressures:</p>
      <div className="overflow-x-auto">
        <table className="min-w-full bg-white">
          <thead>
            <tr>
              <th className="py-2 px-4 border-b">Index</th>
              <th className="py-2 px-4 border-b">Pressure (cmH2O)</th>
            </tr>
          </thead>
          <tbody>
            {predictions.slice(0, 10).map((pressure, index) => ( // Display first 10 for brevity
              <tr key={index}>
                <td className="py-2 px-4 border-b">{index + 1}</td>
                <td className="py-2 px-4 border-b">{pressure.toFixed(2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="text-gray-600 mt-4">Showing first 10 predictions. Total: {predictions.length}</p>
    </div>
  );
}

export default Result;