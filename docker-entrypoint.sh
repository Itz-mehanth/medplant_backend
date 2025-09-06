#!/bin/bash

# Set strict error handling
set -euo pipefail

# Function to log messages
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Function to download all models
download_models() {
    log "Downloading plant classification models..."
    
    # Create models directory structure if it doesn't exist
    mkdir -p models/binary models/classifiers
    
    # Binary models
    if [[ -n "${PLANT_BINARY_MODEL_URL:-}" ]]; then
        log "Downloading plant binary model..."
        wget -q -O "models/binary/leaf binary.h5" "$PLANT_BINARY_MODEL_URL"
    fi
    
    if [[ -n "${FLOWER_BINARY_MODEL_URL:-}" ]]; then
        log "Downloading flower binary model..."
        wget -q -O "models/binary/flower binary.h5" "$FLOWER_BINARY_MODEL_URL"
    fi
    
    if [[ -n "${FRUIT_BINARY_MODEL_URL:-}" ]]; then
        log "Downloading fruit binary model..."
        wget -q -O "models/binary/fruit binary.h5" "$FRUIT_BINARY_MODEL_URL"
    fi
    
    # Classifier models
    if [[ -n "${PLANT_CLASSIFIER_MODEL_URL:-}" ]]; then
        log "Downloading plant classifier model..."
        wget -q -O "models/classifiers/leaf classifier.keras" "$PLANT_CLASSIFIER_MODEL_URL"
    fi
    
    if [[ -n "${FLOWER_CLASSIFIER_MODEL_URL:-}" ]]; then
        log "Downloading flower classifier model..."
        wget -q -O "models/classifiers/flower classifier.keras" "$FLOWER_CLASSIFIER_MODEL_URL"
    fi
    
    if [[ -n "${FRUIT_CLASSIFIER_MODEL_URL:-}" ]]; then
        log "Downloading fruit classifier model..."
        wget -q -O "models/classifiers/fruit classifier.keras" "$FRUIT_CLASSIFIER_MODEL_URL"
    fi
    
    # Verify models exist
    binary_models=0
    classifier_models=0
    
    for model in "models/binary/leaf binary.h5" "models/binary/flower binary.h5" "models/binary/fruit binary.h5"; do
        if [[ -f "$model" ]]; then
            binary_models=$((binary_models + 1))
            log "Binary model found: $model"
        fi
    done
    
    for model in "models/classifiers/leaf classifier.keras" "models/classifiers/flower classifier.keras" "models/classifiers/fruit classifier.keras"; do
        if [[ -f "$model" ]]; then
            classifier_models=$((classifier_models + 1))
            log "Classifier model found: $model"
        fi
    done
    
    if [[ $binary_models -eq 0 ]] && [[ $classifier_models -eq 0 ]]; then
        log "WARNING: No model files found. App will start but predictions may fail."
    else
        log "Model download completed. Found $binary_models binary models and $classifier_models classifier models."
    fi
}

# Function to validate environment variables
validate_env() {
    log "Validating environment variables..."
    
    # Check for model URLs
    model_urls=("PLANT_BINARY_MODEL_URL" "FLOWER_BINARY_MODEL_URL" "FRUIT_BINARY_MODEL_URL" 
                "PLANT_CLASSIFIER_MODEL_URL" "FLOWER_CLASSIFIER_MODEL_URL" "FRUIT_CLASSIFIER_MODEL_URL")
    found_urls=0
    
    for var in "${model_urls[@]}"; do
        if [[ -n "${!var:-}" ]]; then
            found_urls=$((found_urls + 1))
        fi
    done
    
    if [[ $found_urls -eq 0 ]]; then
        log "WARNING: No model URLs provided. Make sure model files exist in the models/ directory."
    else
        log "Found $found_urls model URLs"
    fi
    
    log "Environment validation completed"
}

# Function to wait for dependencies
wait_for_dependencies() {
    log "Checking if models directories exist..."
    
    if [[ ! -d "models/binary" ]]; then
        mkdir -p models/binary
        log "Created models/binary directory"
    fi
    
    if [[ ! -d "models/classifiers" ]]; then
        mkdir -p models/classifiers
        log "Created models/classifiers directory"
    fi
    
    log "Dependencies ready"
}

# Function to run health check
health_check() {
    log "Running initial health check..."
    
    # Start the app in background for health check
    python app.py &
    APP_PID=$!
    
    # Wait for app to start
    sleep 15
    
    # Check if app is responding
    if curl -f http://localhost:${PORT:-5000}/health > /dev/null 2>&1; then
        log "Health check passed"
    else
        log "Health check failed - but continuing anyway (models might still be loading)"
    fi
    
    # Stop the background app
    kill $APP_PID 2>/dev/null || true
    wait $APP_PID 2>/dev/null || true
}

# Function to start the application
start_app() {
    log "Starting the plant classification application..."
    
    # Set the port from environment variable or default to 5000
    PORT=${PORT:-5000}
    
    # Check if gunicorn is available, otherwise use Flask dev server
    if command -v gunicorn > /dev/null 2>&1; then
        log "Starting with Gunicorn..."
        exec gunicorn \
            --bind 0.0.0.0:$PORT \
            --workers ${WORKERS:-1} \
            --worker-class eventlet \
            --worker-connections 1000 \
            --max-requests 1000 \
            --max-requests-jitter 100 \
            --timeout 300 \
            --keep-alive 2 \
            --log-level info \
            --access-logfile - \
            --error-logfile - \
            --preload \
            app:app
    else
        log "Gunicorn not found, starting with Flask dev server..."
        export FLASK_APP=app.py
        export FLASK_ENV=production
        exec python app.py
    fi
}

# Main execution
main() {
    log "Starting Docker entrypoint for plant classification..."
    
    # Validate environment variables
    validate_env
    
    # Download models if URLs are provided
    download_models
    
    # Wait for dependencies
    wait_for_dependencies
    
    # Run initial health check (optional, uncomment if needed)
    # health_check
    
    # Start the application
    start_app
}

# Handle signals gracefully
cleanup() {
    log "Received shutdown signal, cleaning up..."
    # Kill any background processes
    jobs -p | xargs -r kill 2>/dev/null || true
    exit 0
}

trap cleanup SIGTERM SIGINT

# Run main function
main "$@"