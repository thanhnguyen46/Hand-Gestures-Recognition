import React, { useRef, useEffect, useState } from 'react';
import Webcam from 'react-webcam';
import * as tf from '@tensorflow/tfjs';
import * as handpose from '@tensorflow-models/handpose';
import './App.css';

function App() {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const [gestureClass, setGestureClass] = useState('');

  useEffect(() => {
    const modelPath = '/handpose_model/model.json';
            
    const loadModel = async () => {
      try {
        const model = await tf.loadLayersModel(modelPath);
        console.log('Trained model loaded.');
        return model;
      } catch (error) {
        console.error('Error loading model:', error);
        throw error;
      }
    };

    const runHandpose = async () => {
      const net = await handpose.load();
      console.log('Handpose model loaded.');

      const model = await loadModel();

      setInterval(() => {
        detect(net, model);
      }, 100);
    };

    const detect = async (net, model) => {
      if (
        typeof webcamRef.current !== 'undefined' &&
        webcamRef.current !== null &&
        webcamRef.current.video.readyState === 4
      ) {
        const video = webcamRef.current.video;
        const videoWidth = video.videoWidth;
        const videoHeight = video.videoHeight;

        webcamRef.current.video.width = videoWidth;
        webcamRef.current.video.height = videoHeight;

        canvasRef.current.width = videoWidth;
        canvasRef.current.height = videoHeight;

        const hand = await net.estimateHands(video);

        const ctx = canvasRef.current.getContext('2d');
        ctx.clearRect(0, 0, videoWidth, videoHeight);

        if (hand.length > 0) {
          const landmarks = hand[0].landmarks;
          const minX = Math.min(...landmarks.map(l => l[0])) - 20;
          const minY = Math.min(...landmarks.map(l => l[1])) - 20;
          const maxX = Math.max(...landmarks.map(l => l[0])) + 20;
          const maxY = Math.max(...landmarks.map(l => l[1])) + 20;

          ctx.strokeStyle = 'red';
          ctx.lineWidth = 2;
          ctx.strokeRect(minX, minY, maxX - minX, maxY - minY);

          // Perform gesture classification
          const handImage = ctx.getImageData(minX, minY, maxX - minX, maxY - minY);
          const tensor = tf.browser.fromPixels(handImage, 1)
            .resizeNearestNeighbor([240, 640])
            .toFloat()
            .div(255.0)
            .expandDims();
          const prediction = model.predict(tensor);
          const classIndex = prediction.argMax(1).dataSync()[0];
          const gestureClasses = ['palm', 'l', 'fist', 'fist_moved', 'thumb', 'index', 'ok', 'palm_moved', 'c', 'down'];
          setGestureClass(gestureClasses[classIndex]);
        } else {
          setGestureClass('');
        }
      }
    };

    runHandpose();
  }, []);

  return (
    <div className="App">
      <h1>Gesture Recognition</h1>
      <div className="webcam-container">
        <Webcam ref={webcamRef} style={{ width: '100%', height: 'auto' }} />
        <canvas ref={canvasRef} className="canvas-overlay" />
        {gestureClass && <div className="gesture-class">{gestureClass}</div>}
      </div>
    </div>
  );
}

export default App;