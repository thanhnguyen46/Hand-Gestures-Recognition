from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes and origins

# Load the model architecture from the HDF5 file
model = tf.keras.models.load_model('GestureRecognition_model.h5')

# Load the model weights from the separate weights file
model.load_weights('GestureRecognition.weights.h5')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image data from the request
        image_data = request.json['image']
        
        # Preprocess the image data
        image_array = np.array(image_data, dtype=np.float32) / 255.0
        image_array = np.reshape(image_array, (240, 640, 1))  # Reshape the input data
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension   
             
        # Make predictions using the loaded model
        predictions = model.predict(image_array)
    
        # Get the predicted class index
        class_index = np.argmax(predictions[0])
        
        # Define the gesture classes
        gesture_classes = ['palm', 'l', 'fist', 'fist_moved', 'thumb', 'index', 'ok', 'palm_moved', 'c', 'down']
        
        # Get the predicted gesture class
        predicted_class = gesture_classes[class_index]
    
        return jsonify({'gesture': predicted_class})
    
    except Exception as e:
        app.logger.error(f"Error during prediction: {str(e)}")
        return jsonify({'error': 'An error occurred during prediction'}), 500

if __name__ == '__main__':
    app.run(port=5523)