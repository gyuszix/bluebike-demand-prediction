#!/bin/bash

# === CONFIG ===
SERVICE_NAME="bluebikes-backend"
PROJECT_ID="bluebike-demo-gyuszix"
REGION="us-central1"
IMAGE="gcr.io/$PROJECT_ID/$SERVICE_NAME"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Building Docker image..."
docker build --platform linux/amd64 -t $IMAGE -f "$SCRIPT_DIR/Dockerfile" "$SCRIPT_DIR"

if [ $? -ne 0 ]; then
  echo "Docker build failed."
  exit 1
fi

echo "Pushing image to GCR..."
docker push $IMAGE

echo "Deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
  --image $IMAGE \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated

echo "Backend deployment complete."
