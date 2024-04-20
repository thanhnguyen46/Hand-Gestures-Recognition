import React, { useRef, useEffect, useState } from 'react';
import Webcam from 'react-webcam';
import * as tf from '@tensorflow/tfjs';
import * as handpose from '@tensorflow-models/handpose';
import axios from 'axios';
import './App.css';

function App() {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const [gestureClass, setGestureClass] = useState('');

  useEffect(() => {
    const runHandpose = async () => {
      const net = await handpose.load({
        modelUrl: '/handpose_model/model.json',
        inputResolution: { width: 640, height: 480 },
        scale: 0.8
      });
      console.log('Handpose model loaded.');
      setInterval(() => {
        detect(net);
      }, 100);
    };

    const detect = async (net) => {
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
            .resizeNearestNeighbor([240, 640]) // Resize the image to (240, 640)
            .toFloat()
            .div(255.0)
            .expandDims();

          // Send image data to the backend for prediction
          const response = await axios.post('http://localhost:5523/predict', {
            image: Array.from(tensor.dataSync())
          });

          setGestureClass(response.data.gesture);
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