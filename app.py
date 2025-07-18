import os
import json
import numpy as np
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline
import cv2
from sentence_transformers import SentenceTransformer
import faiss
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficient_preprocess
from tensorflow.keras.applications.mobilenet import preprocess_input as mobile_preprocess
import firebase_admin
from firebase_admin import credentials, firestore
from geopy.geocoders import Nominatim
from google.cloud.firestore_v1 import GeoPoint
import tempfile
import logging
import signal
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": 
                     {"origins": 
                      ["http://localhost:3000",
                       "https://medicinal-plant-82aa9.web.app"]
                    }})

# Global variables
plant_data = []
model_st = None
index = None
models = {}
model_lock = threading.Lock()

# Thread pool for concurrent operations
executor = ThreadPoolExecutor(max_workers=4)

# Class names
leaf_class_names = ['Aloevera', 'Amla', 'Amruthaballi', 'Arali', 'Astma_weed']
fruit_class_names = ['Lemon', 'Mango', 'Papaya', 'Pomoegranate', 'Tomato']

# Model configurations
MODEL_CONFIGS = {
    'leaf': {
        'models': ['leaf_model', 'eff_leaf_model', 'mob_leaf_model'],
        'class_names': leaf_class_names,
        'weights': [0.4, 0.3, 0.3]
    },
    'fruit': {
        'models': ['eff_fruit_model', 'mob_fruit_model'],
        'class_names': fruit_class_names,
        'weights': [0.5, 0.5]
    }
}

class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, GeoPoint):
            return {"latitude": obj.latitude, "longitude": obj.longitude}
        return super().default(obj)

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    logger.info(f"Received signal {signum}, shutting down gracefully...")
    executor.shutdown(wait=False)
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

def initialize_firebase():
    """Initialize Firebase Admin SDK"""
    try:
        if not firebase_admin._apps:
            cred_path = 'serviceAccountKey.json'
            if cred_path and os.path.exists(cred_path):
                cred = credentials.Certificate(cred_path)
                firebase_admin.initialize_app(cred)
                logger.info("Firebase initialized successfully")
            else:
                logger.error("Firebase credentials file not found")
                return None
        
        return firestore.client()
    except Exception as e:
        logger.error(f"Firebase initialization error: {str(e)}")
        return None

def get_address_from_coordinates(lat, lng):
    """Get address from coordinates using Nominatim"""
    try:
        geolocator = Nominatim(user_agent="plantApp")
        location = geolocator.reverse((lat, lng), language='en', timeout=10)
        if location:
            return location.address
        else:
            return "Address not found"
    except Exception as e:
        logger.error(f"Geocoding error: {str(e)}")
        return "Address not available"

def fetch_plant_data():
    """Fetch plant data from Firestore"""
    try:
        db = initialize_firebase()
        if not db:
            logger.warning("Firebase not available, using empty plant data")
            return []
        
        collection_ref = db.collection('plant_details')
        docs = collection_ref.stream()
        
        plant_details = []
        
        for doc in docs:
            plant_info = doc.to_dict()
            
            # Fetch images collection
            images = []
            try:
                images_ref = doc.reference.collection('images').stream()
                for image in images_ref:
                    image_data = image.to_dict()
                    if 'url' in image_data:
                        images.append(image_data['url'])
            except Exception as e:
                logger.error(f"Error fetching images: {str(e)}")
            
            # Fetch coordinates collection
            coordinates = []
            try:
                coordinates_ref = doc.reference.collection('coordinates').stream()
                for coord in coordinates_ref:
                    coord_data = coord.to_dict()
                    geo_point = coord_data.get('location')
                    if geo_point and isinstance(geo_point, GeoPoint):
                        lat, lng = geo_point.latitude, geo_point.longitude
                        address = get_address_from_coordinates(lat, lng)
                        coordinates.append({
                            'coordinates': geo_point, 
                            'address': address
                        })
            except Exception as e:
                logger.error(f"Error fetching coordinates: {str(e)}")
            
            plant_info['images'] = images
            plant_info['coordinates'] = coordinates
            plant_details.append(plant_info)
        
        logger.info(f"Fetched {len(plant_details)} plants from Firestore")
        return plant_details
    except Exception as e:
        logger.error(f"Error fetching plant data: {str(e)}")
        return []

def format_plant_data(plant):
    """Format plant data for embedding"""
    family = plant.get("Family", "No family provided")
    description = plant.get("Description", "No description available")
    scientific_name = plant.get("Scientific Name", "No scientific name provided")
    common_name = plant.get("Common Name", "No common name provided")
    
    coordinates = plant.get("coordinates", [])
    if coordinates:
        coord_info = coordinates[0]
        address = coord_info.get("address", "No address available")
        lat_lon = coord_info.get("coordinates", None)
        
        if isinstance(lat_lon, GeoPoint):
            latitude = lat_lon.latitude
            longitude = lat_lon.longitude
        elif isinstance(lat_lon, dict):
            latitude = lat_lon.get("latitude", "No latitude")
            longitude = lat_lon.get("longitude", "No longitude")
        else:
            latitude = "No latitude"
            longitude = "No longitude"
        
        location_info = f"Located at {latitude}, {longitude}, Address: {address}"
    else:
        location_info = "No location data available"
    
    return f"Family: {family}. Scientific Name: {scientific_name}. Common Name: {common_name}. Description: {description}. {location_info}"

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
    global model_st, index, plant_data, models
    
    try:
        # Initialize SentenceTransformer
        logger.info("Initializing SentenceTransformer...")
        model_st = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Fetch plant data
        logger.info("Fetching plant data...")
        plant_data = fetch_plant_data()
        
        if plant_data:
            # Create embeddings
            logger.info("Creating embeddings...")
            plant_descriptions = [format_plant_data(plant) for plant in plant_data]
            plant_embeddings = model_st.encode(plant_descriptions)
            
            # Initialize FAISS index
            dimension = len(plant_embeddings[0])
            index = faiss.IndexFlatL2(dimension)
            index.add(np.array(plant_embeddings))
            
            logger.info(f"Initialized embeddings for {len(plant_data)} plants")
        else:
            logger.warning("No plant data available for embeddings")
        
        # Load ML models
        logger.info("Loading ML models...")
        model_paths = {
            'leaf_model': 'models/LEAF_model.h5',
            'eff_leaf_model': 'models/LEAF_model_eff.h5',
            'mob_leaf_model': 'models/LEAF_model_mobilenet.h5',
            'fruit_model': 'models/FRUIT_model.h5',
            'eff_fruit_model': 'models/FRUIT_model_eff.h5',
            'mob_fruit_model': 'models/FRUIT_model_mobilenet.h5'
        }
        
        # Load models concurrently
        with ThreadPoolExecutor(max_workers=3) as model_executor:
            futures = {}
            for model_name, model_path in model_paths.items():
                future = model_executor.submit(load_model_safe, model_path, model_name)
                futures[model_name] = future
            
            # Collect results
            for model_name, future in futures.items():
                models[model_name] = future.result()
        
        # Log loaded models
        loaded_models = [name for name, model in models.items() if model is not None]
        logger.info(f"Loaded models: {loaded_models}")
        
        logger.info("Model initialization completed successfully")
        
    except Exception as e:
        logger.error(f"Error initializing models: {str(e)}")

def get_location_info(plant):
    """Get location information from plant data"""
    coordinates = plant.get('coordinates', [])
    
    if not coordinates:
        return "No location data available"
    
    coord_info = coordinates[0]
    geo_point = coord_info.get('coordinates')
    address = coord_info.get('address', 'No address available')
    
    if isinstance(geo_point, GeoPoint):
        latitude = geo_point.latitude
        longitude = geo_point.longitude
        return f"Lat: {latitude}, Lon: {longitude}, Address: {address}"
    elif isinstance(geo_point, dict):
        latitude = geo_point.get('latitude', 'No latitude')
        longitude = geo_point.get('longitude', 'No longitude')
        return f"Lat: {latitude}, Lon: {longitude}, Address: {address}"
    
    return "Invalid coordinates format"

def generate_answer(query, retrieved_plants, max_plants=5):
    """Generate answer using Hugging Face API"""
    try:
        context = "\n\n".join([
            f"Family: {plant.get('Family', 'N/A')}. "
            f"Scientific Name: {plant.get('Scientific Name', 'N/A')}. "
            f"Common Name: {plant.get('Common Name', 'N/A')}. "
            f"Description: {plant.get('Description', 'No description available')}. "
            f"Location: {get_location_info(plant)}"
            for plant in retrieved_plants[:max_plants]
        ])
        
        input_text = f"Query: {query}\n\nContext:\n{context}\n\nPlease provide the relevant information"
        
        API_URL = "https://api-inference.huggingface.co/models/Qwen/Qwen2.5-Coder-32B-Instruct"
        headers = {"Authorization": f"Bearer {os.getenv('HUGGINGFACE_API_TOKEN')}"}
        
        data = {
            "inputs": input_text,
            "parameters": {"max_new_tokens": 250},
            "task": "text-generation"
        }

        response = requests.post(API_URL, headers=headers, json=data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                generated_text = result[0].get("generated_text", "")
                # Remove the input text from the response
                if generated_text.startswith(input_text):
                    generated_text = generated_text[len(input_text):].strip()
                return generated_text
            else:
                return "I'm sorry, but I couldn't process your request at this time."
        else:
            logger.error(f"API Error {response.status_code}: {response.text}")
            return "I'm sorry, but I couldn't process your request at this time."
    
    except Exception as e:
        logger.error(f"Error generating answer: {str(e)}")
        return "I'm sorry, but I encountered an error while processing your request."

def retrieve_documents(query, top_k=3):
    """Retrieve relevant documents using FAISS"""
    try:
        if not model_st or not index:
            logger.warning("Model or index not available for document retrieval")
            return []
        
        query_embedding = model_st.encode([query])
        distances, indices = index.search(np.array(query_embedding), top_k)
        return [plant_data[idx] for idx in indices[0] if idx < len(plant_data)]

    except Exception as e:
        logger.error(f"Error retrieving documents: {str(e)}")
        return []

def download_image(img_url):
    """Download image from URL"""
    try:
        response = requests.get(img_url, stream=True, timeout=30)
        response.raise_for_status()

        # Create a temporary file
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
        # Check if critical components are loaded
        health_status = {
            'status': 'healthy',
            'models_loaded': len([k for k, v in models.items() if v is not None]),
            'embeddings_ready': model_st is not None,
            'plant_data_count': len(plant_data),
            'timestamp': str(np.datetime64('now'))
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
        
        if detection_type not in ['leaf', 'fruit']:
            return jsonify({'error': 'Invalid type. Use "leaf" or "fruit"'}), 400
        
        # Get model configuration
        config = MODEL_CONFIGS[detection_type]
        model_names = config['models']
        class_names = config['class_names']
        weights = config['weights']
        
        # Check if any models are loaded
        available_models = [name for name in model_names if models.get(name) is not None]
        if not available_models:
            return jsonify({'error': f'{detection_type.title()} models not available'}), 503
        
        # Download and preprocess image
        image_path = download_image(img_url)
        
        # Preprocess for different models
        preprocess_funcs = [resnet_preprocess, efficient_preprocess, mobile_preprocess]
        processed_images = []
        
        for preprocess_func in preprocess_funcs:
            processed_image = preprocess_image_for_model(image_path, preprocess_func)
            processed_images.append(processed_image)
        
        predictions = []
        model_results = {}
        
        # Make predictions with available models
        with model_lock:
            for i, model_name in enumerate(model_names):
                if models.get(model_name) is not None:
                    try:
                        pred = models[model_name].predict(processed_images[i])
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
            'model_results': model_results,
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

@app.route('/get_plant_info', methods=['POST'])
def get_plant_info():
    """Plant information retrieval endpoint"""
    try:
        data = request.get_json()
        query = data.get('query', None)
        
        if not query:
            return jsonify({'error': 'Query is missing'}), 400
        
        if not plant_data:
            return jsonify({'error': 'Plant database not available'}), 503
        
        instruction = (
            "You are an AI assistant embedded in a plant information app. "
            "If the query is not related to plants, just respond that you can't answer them. "
            "Respond concisely to user queries about plants using the database provided. "
            "Do not respond with unrelated suggestions or conversational fillers. "
            "Answer the query strictly based on the query's intent."
        )
        
        full_query = f"{instruction} {query}"
        
        # Retrieve relevant plants
        retrieved_plants = retrieve_documents(full_query)
        
        if not retrieved_plants:
            return jsonify({
                'query': query,
                'answer': "I couldn't find any relevant plant information for your query.",
                'retrieved_count': 0
            }), 200
        
        # Generate answer
        generated_answer = generate_answer(full_query, retrieved_plants)
        
        return jsonify({
            'query': query,
            'answer': generated_answer,
            'retrieved_count': len(retrieved_plants)
        }), 200
    
    except Exception as e:
        logger.error(f"Plant info error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get API statistics"""
    try:
        stats = {
            'total_plants': len(plant_data),
            'models_loaded': {name: model is not None for name, model in models.items()},
            'embeddings_ready': model_st is not None,
            'faiss_index_ready': index is not None
        }
        return jsonify(stats), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Initialize models when the app starts
logger.info("Starting application initialization...")
initialize_models()
logger.info("Application initialization completed")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)