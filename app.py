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
CORS(app, resources={r"/*": {"origins": ["http://localhost:3000", "https://medicinal-plant-82aa9.web.app"]}})

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
        
        # Load models sequentially to avoid memory issues
        for model_name, model_path in model_paths.items():
            models[model_name] = load_model_safe(model_path, model_name)
        
        # Log loaded models
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

        # Resize and preprocess
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

@app.route('/predict', methods=['POST'])
def predict():
    """Plant prediction endpoint"""
    image_path = None
    try:
        data = request.get_json()
        if not data or 'image_url' not in data or 'type' not in data:
            return jsonify({'error': 'Parameters "image_url" and "type" are required'}), 400
        
        img_url = data.get('image_url')
        detection_type = data.get('type').lower()
        
        if detection_type not in MODEL_CONFIGS:
            return jsonify({'error': f'Invalid type. Use one of: {list(MODEL_CONFIGS.keys())}'}), 400
        
        # Get model configuration
        config = MODEL_CONFIGS[detection_type]
        model_names = config['models']
        class_names = config['class_names']
        weights = config['weights']
        preprocess_funcs = config['preprocess_funcs']
        
        # Check if any models are loaded
        available_models = [name for name in model_names if models.get(name) is not None]
        if not available_models:
            return jsonify({'error': f'{detection_type.title()} models not available'}), 503
        
        # Download image
        image_path = download_image(img_url)
        
        predictions = []
        model_results = {}
        
        # Make predictions with available models
        with model_lock:
            for i, model_name in enumerate(model_names):
                if models.get(model_name) is not None:
                    try:
                        # Preprocess image for this specific model
                        processed_image = preprocess_image_for_model(image_path, preprocess_funcs[i])
                        pred = models[model_name].predict(processed_image, verbose=0)
                        predictions.append(pred[0])
                        model_results[model_name] = pred[0].tolist()
                    except Exception as e:
                        logger.error(f"Error predicting with {model_name}: {str(e)}")
        
        if not predictions:
            return jsonify({'error': 'No models available for prediction'}), 503
        
        # Ensemble predictions (weighted average)
        if len(predictions) == len(weights):
            combined_probabilities = np.average(predictions, axis=0, weights=weights)
        else:
            combined_probabilities = np.mean(predictions, axis=0)
        
        # Get final prediction
        predicted_class_index = np.argmax(combined_probabilities)
        predicted_class_name = class_names[predicted_class_index]
        confidence_level = float(combined_probabilities[predicted_class_index])
        
        # Prepare response
        response = {
            'type': detection_type,
            'predicted_class': predicted_class_name,
            'confidence_level': confidence_level,
            'class_probabilities': {
                name: float(prob) for name, prob in zip(class_names, combined_probabilities)
            },
            'models_used': available_models
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500
    
    finally:
        # Clean up temporary file
        if image_path:
            try:
                os.unlink(image_path)
            except:
                pass

# Initialize models when the module is imported
initialize_models()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)