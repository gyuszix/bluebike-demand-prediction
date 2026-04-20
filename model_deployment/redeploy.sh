#!/bin/bash
echo "Redeploying with latest changes"

# Rebuild and push
docker build -t gcr.io/bluebike-demo-gyuszix/bluebikes-prediction:latest . && \
docker push gcr.io/bluebike-demo-gyuszix/bluebikes-prediction:latest && \

# Deploy
gcloud run deploy bluebikes-prediction \
  --image gcr.io/bluebike-demo-gyuszix/bluebikes-prediction:latest \
  --region us-central1 \
  --allow-unauthenticated \
  --min-instances 0 \
  --project bluebike-demo-gyuszix

echo "Deployment complete!"