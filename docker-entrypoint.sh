#!/bin/bash

# Set strict error handling
set -euo pipefail

# Function to log messages
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Function to download models
download_models() {
    log "Downloading models..."
    
    # Create models directory if it doesn't exist
    mkdir -p models
    
    # Download models from URLs if provided
    if [[ -n "${LEAF_MODEL_URL:-}" ]]; then
        log "Downloading leaf model..."
        wget -q -O models/LEAF_model.h5 "$LEAF_MODEL_URL"
    fi
    
    if [[ -n "${EFF_LEAF_MODEL_URL:-}" ]]; then
        log "Downloading efficient leaf model..."
        wget -q -O models/LEAF_model_eff.h5 "$EFF_LEAF_MODEL_URL"
    fi
    
    if [[ -n "${MOB_LEAF_MODEL_URL:-}" ]]; then
        log "Downloading mobile leaf model..."
        wget -q -O models/LEAF_model_mobilenet.h5 "$MOB_LEAF_MODEL_URL"
    fi
    
    if [[ -n "${FRUIT_MODEL_URL:-}" ]]; then
        log "Downloading fruit model..."
        wget -q -O models/FRUIT_model.h5 "$FRUIT_MODEL_URL"
    fi
    
    if [[ -n "${EFF_FRUIT_MODEL_URL:-}" ]]; then
        log "Downloading efficient fruit model..."
        wget -q -O models/FRUIT_model_eff.h5 "$EFF_FRUIT_MODEL_URL"
    fi
    
    if [[ -n "${MOB_FRUIT_MODEL_URL:-}" ]]; then
        log "Downloading mobile fruit model..."
        wget -q -O models/FRUIT_model_mobilenet.h5 "$MOB_FRUIT_MODEL_URL"
    fi
    
    log "Model download completed"
}

# Function to validate environment variables
validate_env() {
    log "Validating environment variables..."
    
    # Check required environment variables
    required_vars=("HUGGINGFACE_API_TOKEN" "FIREBASE_SERVICE_ACCOUNT_JSON")
    
    for var in "${required_vars[@]}"; do
        if [[ -z "${!var:-}" ]]; then
            log "ERROR: Required environment variable $var is not set"
            exit 1
        fi
    done
    
    log "Environment validation completed"
}

# Function to setup Firebase credentials
setup_firebase() {
    log "Setting up Firebase credentials..."
    
    # Write Firebase service account JSON to file
    echo "$FIREBASE_SERVICE_ACCOUNT_JSON" > /tmp/firebase-service-account.json
    export GOOGLE_APPLICATION_CREDENTIALS=/tmp/firebase-service-account.json
    
    log "Firebase credentials setup completed"
}

# Function to wait for dependencies
wait_for_dependencies() {
    log "Waiting for dependencies to be ready..."
    
    # Add any dependency checks here
    # For example, waiting for database connections
    
    log "Dependencies ready"
}

# Function to run health check
health_check() {
    log "Running initial health check..."
    
    # Start the app in background for health check
    python app.py &
    APP_PID=$!
    
    # Wait for app to start
    sleep 10
    
    # Check if app is responding
    if curl -f http://localhost:5000/health > /dev/null 2>&1; then
        log "Health check passed"
    else
        log "Health check failed"
        kill $APP_PID 2>/dev/null || true
        exit 1
    fi
    
    # Stop the background app
    kill $APP_PID 2>/dev/null || true
    wait $APP_PID 2>/dev/null || true
}

# Function to start the application
start_app() {
    log "Starting the application..."
    
    # Set the port from environment variable or default to 5000
    PORT=${PORT:-5000}
    
    # Start the application with gunicorn
    exec gunicorn \
        --bind 0.0.0.0:$PORT \
        --workers ${WORKERS:-2} \
        --worker-class sync \
        --worker-connections 1000 \
        --max-requests 1000 \
        --max-requests-jitter 100 \
        --timeout 120 \
        --keep-alive 2 \
        --log-level info \
        --access-logfile - \
        --error-logfile - \
        app:app
}

# Main execution
main() {
    log "Starting Docker entrypoint..."
    
    # Validate environment variables
    validate_env
    
    # Setup Firebase
    setup_firebase
    
    # Download models if URLs are provided
    download_models
    
    # Wait for dependencies
    wait_for_dependencies
    
    # Run initial health check (optional, comment out if not needed)
    # health_check
    
    # Start the application
    start_app
}

# Handle signals
trap 'log "Received SIGTERM, shutting down gracefully..."; exit 0' SIGTERM
trap 'log "Received SIGINT, shutting down gracefully..."; exit 0' SIGINT

# Run main function
main "$@"