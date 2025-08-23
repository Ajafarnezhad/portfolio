import React, { useState, useRef } from 'react';

function Dropzone({ onPrediction, onError }) {
  const [fileName, setFileName] = useState('Drag and drop a CSV file here or click to select');
  const fileInputRef = useRef(null);

  const handleDragOver = (e) => {
    e.preventDefault();
    e.currentTarget.classList.add('border-indigo-500');
  };

  const handleDragLeave = (e) => {
    e.currentTarget.classList.remove('border-indigo-500');
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.currentTarget.classList.remove('border-indigo-500');
    const file = e.dataTransfer.files[0];
    if (file) {
      setFileName(file.name);
      handleUpload(file);
    }
  };

  const handleChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setFileName(file.name);
      handleUpload(file);
    }
  };

  const handleUpload = async (file) => {
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      if (response.ok) {
        onPrediction(data.predictions);
      } else {
        onError(data.error || 'Failed to process file');
      }
    } catch (error) {
      onError('Network error. Please try again.');
    }
  };

  return (
    <div className="bg-white shadow-lg rounded-lg p-6">
      <h2 className="text-xl font-semibold mb-4">Upload a CSV File</h2>
      <div
        className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center cursor-pointer hover:border-indigo-500 transition"
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current.click()}
      >
        <input
          type="file"
          id="file"
          name="file"
          accept=".csv"
          className="hidden"
          ref={fileInputRef}
          onChange={handleChange}
        />
        <p className="text-gray-500">{fileName}</p>
      </div>
    </div>
  );
}

export default Dropzone;