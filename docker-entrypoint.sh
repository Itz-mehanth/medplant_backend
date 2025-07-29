#!/bin/bash

# Set strict error handling
set -euo pipefail

# Function to log messages
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Function to download leaf models only
download_models() {
    log "Downloading leaf models..."
    
    # Create models directory if it doesn't exist
    mkdir -p models
    
    # Download only leaf models (matching the exact filenames from your app)
    if [[ -n "${LEAF_MODEL_URL:-}" ]]; then
        log "Downloading leaf model..."
        wget -q -O "models/LEAF model.h5" "$LEAF_MODEL_URL"
    fi
    
    if [[ -n "${EFF_LEAF_MODEL_URL:-}" ]]; then
        log "Downloading efficient leaf model..."
        wget -q -O "models/LEAF model(eff).h5" "$EFF_LEAF_MODEL_URL"
    fi
    
    if [[ -n "${MOB_LEAF_MODEL_URL:-}" ]]; then
        log "Downloading mobile leaf model..."
        wget -q -O "models/LEAF model(mobilenet).h5" "$MOB_LEAF_MODEL_URL"
    fi
    
    # Verify models exist
    models_count=0
    for model in "models/LEAF model.h5" "models/LEAF model(eff).h5" "models/LEAF model(mobilenet).h5"; do
        if [[ -f "$model" ]]; then
            models_count=$((models_count + 1))
            log "Model found: $model"
        fi
    done
    
    if [[ $models_count -eq 0 ]]; then
        log "WARNING: No model files found. App will start but predictions may fail."
    else
        log "Model download completed. Found $models_count models."
    fi
}

# Function to validate environment variables (simplified)
validate_env() {
    log "Validating environment variables..."
    
    # Only check for model URLs if they're needed
    model_urls=("LEAF_MODEL_URL" "EFF_LEAF_MODEL_URL" "MOB_LEAF_MODEL_URL")
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

# Function to wait for dependencies (simplified)
wait_for_dependencies() {
    log "Checking if models directory exists..."
    
    if [[ ! -d "models" ]]; then
        mkdir -p models
        log "Created models directory"
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
    log "Starting the leaf inference application..."
    
    # Set the port from environment variable or default to 5000
    PORT=${PORT:-5000}
    
    # Check if gunicorn is available, otherwise use Flask dev server
    if command -v gunicorn > /dev/null 2>&1; then
        log "Starting with Gunicorn..."
        exec gunicorn \
            --bind 0.0.0.0:$PORT \
            --workers ${WORKERS:-1} \
            --worker-class sync \
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
    log "Starting Docker entrypoint for leaf inference..."
    
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