import React, { useState } from 'react';

const ResamplingStrategyForm = () => {
  const [file, setFile] = useState(null);
  const [metric, setMetric] = useState('F1');
  const [classifier, setClassifier] = useState('KNN');
  const [resampling, setResampling] = useState(0);
  const [recommendations, setRecommendations] = useState([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [inputKey, setInputKey] = useState(Date.now());

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleMetricChange = (e) => {
    setMetric(e.target.value);
  };

  const handleClassifierChange = (e) => {
    setClassifier(e.target.value);
  };

  const handleResamplingChange = (e) => {
    setResampling(e.target.value);
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    setIsProcessing(true);
    const formData = new FormData();
    // if (file) {
    //   formData.append('file', file); // Use the File object directly
    // }
    formData.append('file', file); // Assuming 'file' is the file to upload
    formData.append('metric', metric);
    formData.append('classifier', classifier);
    formData.append('no_resampling_methods', resampling);
  
    // fetch('http://localhost:5000/runmodel', {
    fetch('https://metallic-recommendation-system-d699fe44117d.herokuapp.com/runmodel', {
      method: 'POST',
      body: formData,
    })
    .then(response => response.json())
    .then(data => {
      console.log(data); // Logging the data for debugging
      setRecommendations(data.recommendations); // Update state with the recommendations from the backend
      setIsProcessing(false);
      setInputKey(Date.now());
    })
    .catch(error => {console.error('Error:', error);
    setIsProcessing(false);
    setInputKey(Date.now());
  });
};

  return (
    <div className="form-container">
      <h2 className="form-item">Use your data to get the best resampling strategies</h2>
      <form onSubmit={handleSubmit} className="form-item">
        <p>Only .csv files are accepted</p>
        <label>
          Upload a File:
          <input type="file" accept=".csv" onChange={handleFileChange} key={inputKey}/>
        </label>
        <br />
        <label>
          Choose a metric:
          <select value={metric} onChange={handleMetricChange}>
            <option value="F1">F1</option>
            <option value="G-mean">G-mean</option>
            <option value="Accuracy">Accuracy</option>
            <option value="Precision">Precision</option>
            <option value="Recall">Recall</option>
            <option value="AUC-ROC">AUC-ROC</option>
            <option value="AUC-PR">AUC-PR</option>
            <option value="BalancedAccuracy">BalancedAccuracy</option>
            <option value="CWA">CWA</option>
            {/* Add other metric options here */}
          </select>
        </label>
        <br />
        <label>
          Choose a classifier:
          <select value={classifier} onChange={handleClassifierChange}>
            <option value="KNN">KNN</option>
            <option value="DT">DT</option>
            <option value="GNB">GNB</option>
            <option value="SVM">SVM</option>
            <option value="RF">RF</option>
            <option value="GB">GB</option>
            <option value="ADA">ADA</option>
            <option value="CAT">CAT</option>
            {/* Add other classifier options here */}
          </select>
        </label>
        <br />
        <label>
          No. of Resampling methods:
          <select value={resampling} onChange={handleResamplingChange}>
            {[...Array(22).keys()].map((number) => (
                <option key={number} value={number}>{number}</option>
            ))}
        </select>
        </label>
        <br />
        <button type="submit" disabled={isProcessing}>
          {isProcessing ? 'Processing...' : 'Process'}
        </button>
      </form>
      {recommendations.length > 0 && (
        <div className="form-item">
          <h3>The recommended resampling methods are:</h3>
          <ol>
            {recommendations.map((method, index) => (
              <li key={index}>{method}</li>
            ))}
          </ol>
        </div>
      )}
    </div>
  );
};



export default ResamplingStrategyForm;
