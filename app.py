import os
import base64
import glob
import io
import logging
import threading
import numpy as np
from flask import request
import cv2
from PIL import Image
from flask import Flask, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_v2_preprocess

# --- Configuration ---
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Flask & SocketIO Initialization ---
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}) # Allow all origins for SocketIO
socketio = SocketIO(app, cors_allowed_origins="*")

# --- Global Variables ---
models = {
    "binary": {},
    "classifiers": {}
}
model_lock = threading.Lock()

# Class names for the classifier models
CLASSIFIER_CLASS_NAMES = {
    'plant': ["Aloevera", "Apple", "Hibiscus", "Lemon", "Mango", "Neem", "Papaya", "Rose", "Tomato", "Tulsi", "Turmeric"],
    'flower': ['Hibiscus', 'Rose'],
    'fruit': ['Apple', 'Lemon', 'Mango', 'Papaya', 'Tomato']
}

@app.route('/debug/models', methods=['GET'])
def debug_models():
    """Debug endpoint to check model files"""
    model_info = {
        'current_directory': os.getcwd(),
        'binary_dir_exists': os.path.exists('models/binary'),
        'classifiers_dir_exists': os.path.exists('models/classifiers'),
        'binary_files': glob.glob('models/binary/*') if os.path.exists('models/binary') else [],
        'classifier_files': glob.glob('models/classifiers/*') if os.path.exists('models/classifiers') else [],
        'all_model_files': glob.glob('models/**/*', recursive=True) if os.path.exists('models') else [],
        'root_contents': os.listdir('.') if os.path.exists('.') else []
    }
    return jsonify(model_info)

# --- Model Loading ---
def load_all_models():
    """
    Loads all binary and classifier models into the global 'models' dictionary.
    """
    logger.info("Initializing all machine learning models...")
    
    model_paths = {
        "binary": {
            "plant": "models/binary/leaf binary.h5",
            "flower": "models/binary/flower binary.h5",
            "fruit": "models/binary/fruit binary.h5"
        },
        "classifiers": {
            "plant": "models/classifiers/leaf classifier.keras",
            "flower": "models/classifiers/flower classifier.keras",
            "fruit": "models/classifiers/fruit classifier.keras"
        }
    }

    for model_type, paths in model_paths.items():
        for category, path in paths.items():
            if os.path.exists(path):
                try:
                    with model_lock:
                        models[model_type][category] = load_model(path)
                    logger.info(f"Successfully loaded model: {path}")
                except Exception as e:
                    logger.error(f"Error loading model {path}: {e}")
            else:
                logger.warning(f"Model file not found: {path}")

    # Log summary
    loaded_binary = list(models['binary'].keys())
    loaded_classifiers = list(models['classifiers'].keys())
    logger.info(f"Model loading complete. Loaded {len(loaded_binary)} binary models: {loaded_binary}")
    logger.info(f"Model loading complete. Loaded {len(loaded_classifiers)} classifier models: {loaded_classifiers}")

# --- Image Processing ---
def preprocess_frame(base64_string):
    """
    Decodes a base64 string into an image and preprocesses it for model prediction.
    Args:
        base64_string (str): The base64 encoded image string (with data URI header).
    Returns:
        np.ndarray: A preprocessed image ready for the model, or None if processing fails.
    """
    try:
        # Remove the data URI header if present (e.g., "data:image/jpeg;base64,")
        if "," in base64_string:
            base64_string = base64_string.split(',')[1]

        # Decode the base64 string
        img_bytes = base64.b64decode(base64_string)
        img = Image.open(io.BytesIO(img_bytes))
        
        # Convert to RGB if it's not
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        # Resize and convert to numpy array
        img = img.resize((224, 224))
        img_array = np.array(img)
        
        # Add batch dimension and preprocess for MobileNetV2
        img_batch = np.expand_dims(img_array, axis=0)
        preprocessed_img = mobilenet_v2_preprocess(img_batch.astype('float32'))
        
        return preprocessed_img
    except Exception as e:
        logger.error(f"Error preprocessing frame: {e}")
        return None

# --- Health Check Endpoint ---
@app.route('/health', methods=['GET'])
def health_check():
    """Provides the status of loaded models."""
    return jsonify({
        'status': 'healthy',
        'loaded_models': {
            'binary': list(models['binary'].keys()),
            'classifiers': list(models['classifiers'].keys())
        }
    }), 200

# --- WebSocket Event Handlers ---
@socketio.on('connect')
def handle_connect():
    """Handles a new client connection."""
    logger.info(f"Client connected: {request.sid}")
    emit('response', {'message': 'Successfully connected to the server!'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handles a client disconnection."""
    logger.info(f"Client disconnected: {request.sid}")

@socketio.on('frame')
def handle_frame(data):
    """
    Receives a video frame, runs predictions on all models, and emits all results.
    """
    logger.debug(f"Received frame from {request.sid}")
    image_data = data.get('image')

    if not image_data:
        emit('error', {'message': 'No image data received.'})
        return

    processed_image = preprocess_frame(image_data)
    if processed_image is None:
        emit('error', {'message': 'Failed to process image frame.'})
        return
    
    all_predictions = []

    # Iterate through all available categories (plant, flower, fruit)
    for category in CLASSIFIER_CLASS_NAMES.keys():
        binary_model = models['binary'].get(category)
        classifier_model = models['classifiers'].get(category)
        
        # Ensure both models for the category exist before proceeding
        if not (binary_model and classifier_model):
            logger.warning(f"Skipping '{category}' as one or both models are missing.")
            continue
        
        try:
            # --- Run Binary Prediction ---
            with model_lock:
                binary_pred = binary_model.predict(processed_image, verbose=0)[0]
            binary_confidence = binary_pred[0] # Assumes single output neuron
            
            # --- Run Classifier Prediction ---
            class_names = CLASSIFIER_CLASS_NAMES[category]
            with model_lock:
                class_preds = classifier_model.predict(processed_image, verbose=0)[0]
            
            pred_index = np.argmax(class_preds)
            classifier_confidence = class_preds[pred_index]
            predicted_class = class_names[pred_index]
            
            # --- Compile results for this category ---
            result_obj = {
                'type': category,
                'binary_confidence': float(binary_confidence),
                'predicted_class': predicted_class,
                'classifier_confidence': float(classifier_confidence)
            }
            all_predictions.append(result_obj)
            
            logger.info(f"Prediction for '{category}': {predicted_class} ({classifier_confidence:.2f}) | Binary: {binary_confidence:.2f}")

        except Exception as e:
            logger.error(f"Prediction error for category '{category}': {e}")
            emit('error', {'message': f"An error occurred during '{category}' prediction."})

    # --- Emit all collected results back to the client ---
    if all_predictions:
        emit('prediction_result', {'results': all_predictions})
        logger.info(f"Sent {len(all_predictions)} prediction sets to client {request.sid}")

# --- Main Execution ---
if __name__ == '__main__':
    load_all_models()
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting SocketIO server on port {port}")
    # Use 'eventlet' for better performance with WebSockets
    socketio.run(app, host="0.0.0.0", port=port, debug=True, allow_unsafe_werkzeug=True)

