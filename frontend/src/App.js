import React, { useState, useCallback } from 'react';
import axios from 'axios';
import { useDropzone } from 'react-dropzone';
import { Chart as ChartJS, CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend } from 'chart.js';
import { Bar } from 'react-chartjs-2';
import './App.css';
import logo from './logo.png';

const API_URL = process.env.REACT_APP_API_URL || 'http://127.0.0.1:8000';

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

const exampleImages = ['example.png', '0b8bdec9d869.png', '0b64a0a06f9a.png', '0babc12807b2.png'];

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [originalImage, setOriginalImage] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [heatmap, setHeatmap] = useState(null);
  const [chartData, setChartData] = useState(null);

  const resetState = () => {
    setPrediction(null);
    setError(null);
    setHeatmap(null);
    setChartData(null);
  }

  const processFile = useCallback((file) => {
    if (file) {
      setSelectedFile(file);
      setOriginalImage(URL.createObjectURL(file));
      resetState();
    }
  }, []);

  const onDrop = useCallback(acceptedFiles => {
    processFile(acceptedFiles[0]);
  }, [processFile]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({ onDrop, accept: 'image/*' });

  const handleFileChange = (event) => { processFile(event.target.files[0]); };

  const handleExampleClick = async (filename) => {
    try {
        const response = await fetch(`/examples/${filename}`); 
        const blob = await response.blob();
        const file = new File([blob], filename, { type: "image/png" });
        processFile(file);
    } catch (err) {
        setError(`Could not load ${filename}.`);
        console.error(err);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      setError("Please select or drop a file first.");
      return;
    }
    setIsLoading(true);
    resetState();

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await axios.post(`${API_URL}/predict`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setPrediction(response.data);
      setHeatmap(response.data.heatmap_image);
      setChartData({
        labels: ["No DR", "Mild", "Moderate", "Severe", "Proliferative"],
        datasets: [{
          label: 'Model Confidence',
          data: response.data.probabilities,
          backgroundColor: 'rgba(13, 110, 253, 0.7)',
          borderColor: 'rgba(13, 110, 253, 1)',
          borderWidth: 1
        }]
      });
    } catch (err) {
      setError("Failed to get a prediction. Is the backend server running?");
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };
  
  const handleExportPDF = async () => {
    const reader = new FileReader();
    reader.readAsDataURL(selectedFile);
    reader.onload = async () => {
        const originalImageBase64 = reader.result.split(',')[1];
        const payload = { ...prediction, original_image: originalImageBase64, heatmap_image: heatmap };
        try {
            // --- FIX: Corrected API call to use the API_URL variable ---
            const response = await axios.post(`${API_URL}/export_pdf`, payload, { responseType: 'blob' });
            const url = window.URL.createObjectURL(new Blob([response.data]));
            const link = document.createElement('a');
            link.href = url;
            link.setAttribute('download', 'AuraSight_Report.pdf');
            document.body.appendChild(link);
            link.click();
            link.remove();
        } catch (err) {
            setError("Failed to generate PDF.");
            console.error(err);
        }
    };
  };

  return (
    <div className="App">
      <nav className="navbar">
        <div className="navbar-brand">
          <img src={logo} alt="AuraSight AI Logo" className="logo"/>
          <h1>AuraSight AI</h1>
        </div>
      </nav>

      <main className="main-content">
        <div className="card">
          <h2 className="card-title">Diabetic Retinopathy Diagnosis</h2>
          <p className="card-subtitle">Upload a retinal scan for an instant, AI-powered analysis.</p>

          <div {...getRootProps({className: `dropzone ${isDragActive ? 'active' : ''}`})}>
            <input {...getInputProps()} onChange={handleFileChange} />
            <p>Drag & drop an image here, or click to select a file</p>
          </div>

          {originalImage && !prediction && (
            <div className="image-preview">
              <h3>Selected Image:</h3>
              <img src={originalImage} alt="Selected Scan Preview" />
            </div>
          )}
          
          <div className="example-gallery">
            <h4>Or try an example:</h4>
            <div className="image-thumbnails">
              {exampleImages.map(filename => (
                <img key={filename} src={`/examples/${filename}`} alt={`Example ${filename}`} onClick={() => handleExampleClick(filename)} />
              ))}
            </div>
          </div>

          <button className="diagnose-button" onClick={handleUpload} disabled={isLoading || !selectedFile}>
            {isLoading ? "Analyzing..." : "Analyze Image"}
          </button>
        </div>

        {error && <p className="error-message">{error}</p>}

        {prediction && (
          <div className="results-card card">
            <div className="results-header">
              <h2 className="card-title">Analysis Report</h2>
              <button className="export-button" onClick={handleExportPDF}>Export PDF</button>
            </div>

            <div className="results-grid">
              <div className="result-item main-diagnosis">
                <h3>Condition</h3>
                <p className="diagnosis-text">{prediction.diagnosis}</p>
                <p className="confidence-text">{prediction.confidence} Confidence</p>
              </div>

              <div className="result-item">
                <h3>Confidence Distribution</h3>
                <div className="chart-container">
                  {chartData && <Bar data={chartData} options={{ indexAxis: 'y', responsive: true, plugins: { legend: { display: false } } }} />}
                </div>
              </div>

              <div className="result-item image-display">
                <h3>Original Scan</h3>
                <img src={originalImage} alt="Original Scan" />
              </div>

              <div className="result-item image-display">
                <h3>AI Focus Heatmap</h3>
                <img src={`data:image/jpeg;base64,${heatmap}`} alt="Grad-CAM Heatmap" />
              </div>
            </div>
          </div>
        )}
      </main>

      <footer className="footer">
        <p>AuraSight AI is an experimental tool. Not for clinical use.</p>
      </footer>
    </div>
  );
}

export default App;
