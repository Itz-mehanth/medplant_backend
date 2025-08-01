import os
import json
import numpy as np
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficient_preprocess
from tensorflow.keras.applications.mobilenet import preprocess_input as mobile_preprocess
import tempfile
import logging
import threading

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
# CORS(app) # Allow all origins for local development
CORS(app, resources={r"/predict": {"origins": "https://medicinal-plant-82aa9.web.app"}})

# Global variables
models = {}
model_lock = threading.Lock()

# Class names
leaf_class_names = ['Aloevera', 'Amla', 'Amruthaballi', 'Arali', 'Astma_weed']

# Model configurations
MODEL_CONFIGS = {
    'leaf': {
        'models': ['leaf_model', 'eff_leaf_model', 'mob_leaf_model'],
        'class_names': leaf_class_names,
        'weights': [0.4, 0.3, 0.3],
        'preprocess_funcs': [resnet_preprocess, efficient_preprocess, mobile_preprocess]
    }
}

def load_model_safe(model_path, model_name):
    """Safely load a model with error handling"""
    try:
        if os.path.exists(model_path):
            model = load_model(model_path)
            logger.info(f"Successfully loaded {model_name}")
            return model
        else:
            logger.warning(f"Model file not found: {model_path}")
            return None
    except Exception as e:
        logger.error(f"Failed to load {model_name}: {str(e)}")
        return None

def initialize_models():
    """Initialize ML models"""
    global models
    
    try:
        logger.info("Loading ML models...")
        model_paths = {
            'leaf_model': 'models/LEAF model.h5',
            'eff_leaf_model': 'models/LEAF model(eff).h5',
            'mob_leaf_model': 'models/LEAF model(mobilenet).h5'
        }
        
        for model_name, model_path in model_paths.items():
            models[model_name] = load_model_safe(model_path, model_name)
        
        loaded_models = [name for name, model in models.items() if model is not None]
        logger.info(f"Loaded models: {loaded_models}")
        
    except Exception as e:
        logger.error(f"Error initializing models: {str(e)}")

def download_image(img_url):
    """Download image from URL"""
    try:
        response = requests.get(img_url, stream=True, timeout=10)
        response.raise_for_status()

        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            for chunk in response.iter_content(chunk_size=8192):
                tmp_file.write(chunk)
            return tmp_file.name

    except Exception as e:
        logger.error(f"Error downloading image: {str(e)}")
        raise ValueError(f"Failed to download image from URL: {img_url}")

def preprocess_image_for_model(image_path, preprocess_func):
    """Preprocess image for model prediction"""
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Failed to load image")

        image = cv2.resize(image, (224, 224))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = preprocess_func(image)
        
        return np.expand_dims(image, axis=0)
    
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        health_status = {
            'status': 'healthy',
            'models_loaded': len([k for k, v in models.items() if v is not None]),
            'available_types': list(MODEL_CONFIGS.keys())
        }
        return jsonify(health_status), 200
    except Exception as e:
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 500

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    """Plant prediction endpoint with enhanced logging"""
    image_path = None
    if request.method == 'GET':
        logger.info("GET request received. Returning welcome message.")
        return jsonify({'message': 'Welcome to the Plant Prediction API! Use POST to make predictions.'}), 200
    try:
        logger.info("Received a prediction request.")
        if 'image' in request.files:
            image_file = request.files['image']
            detection_type = request.form.get('type', 'leaf').lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                image_file.save(tmp_file.name)
                image_path = tmp_file.name
        else:
            data = request.get_json()
            if not data or 'image_url' not in data or 'type' not in data:
                return jsonify({'error': 'Parameters "image_url" and "type" are required for JSON requests'}), 400
            
            img_url = data.get('image_url')
            detection_type = data.get('type').lower()
            image_path = download_image(img_url)

        if detection_type not in MODEL_CONFIGS:
            return jsonify({'error': f'Invalid type. Use one of: {list(MODEL_CONFIGS.keys())}'}), 400
        
        config = MODEL_CONFIGS[detection_type]
        model_names, class_names, weights, preprocess_funcs = config['models'], config['class_names'], config['weights'], config['preprocess_funcs']
        
        available_models = [name for name in model_names if models.get(name) is not None]
        if not available_models:
            return jsonify({'error': f'{detection_type.title()} models not available'}), 503
        
        predictions = []
        model_results = {}
        
        with model_lock:
            for i, model_name in enumerate(model_names):
                if models.get(model_name) is not None:
                    try:
                        processed_image = preprocess_image_for_model(image_path, preprocess_funcs[i])
                        pred = models[model_name].predict(processed_image, verbose=0)
                        predictions.append(pred[0])
                        # ✨ FIX: Store detailed results for each model in the format the UI expects.
                        model_results[model_name] = {
                            "model": model_name,
                            "class_names": class_names,
                            "probabilities": pred[0].tolist()
                        }
                    except Exception as e:
                        logger.error(f"Error predicting with {model_name}: {str(e)}")
        
        if not predictions:
            return jsonify({'error': 'Prediction failed for all models'}), 503
        
        combined_probabilities = np.average(predictions, axis=0, weights=weights) if len(predictions) == len(weights) else np.mean(predictions, axis=0)
        
        predicted_class_index = np.argmax(combined_probabilities)
        predicted_class_name = class_names[predicted_class_index]
        confidence_level = float(combined_probabilities[predicted_class_index])
        
        # ✨ FIX: Build the final response with the detailed 'predictions' map and a 'summary'.
        response = {
            'type': detection_type,
            'predicted_class': predicted_class_name,
            'confidence_level': confidence_level,
            'summary': f"The ensemble model predicts '{predicted_class_name}' with a confidence of {confidence_level:.2%}.",
            'predictions': model_results, # This now contains the detailed breakdown for the UI
            'models_used': available_models
        }
        
        logger.info(f"Prediction successful: {predicted_class_name} with confidence {confidence_level:.2f}")
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"An unexpected error occurred in /predict: {str(e)}")
        return jsonify({'error': 'An internal server error occurred.'}), 500
    
    finally:
        if image_path and os.path.exists(image_path):
            try:
                os.unlink(image_path)
            except Exception as e:
                logger.error(f"Error cleaning up temporary file: {e}")

# Initialize models when the module is imported
initialize_models()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)