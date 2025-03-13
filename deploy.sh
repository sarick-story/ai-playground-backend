#!/bin/bash

# Exit on any error
set -e

# Configuration
PROJECT_ID="employee-managed-validator"
SERVICE_NAME="ai-playground-backend"
REGION="us-central1"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

# Print banner
echo "========================================"
echo "  Deploying ${SERVICE_NAME} to Cloud Run"
echo "========================================"
echo "Project: ${PROJECT_ID}"
echo "Region: ${REGION}"
echo "Image: ${IMAGE_NAME}"
echo "----------------------------------------"

# Step 1: Build and push the Docker image
echo "🔨 Building and pushing Docker image..."
gcloud builds submit --tag ${IMAGE_NAME}

# Step 2: Deploy to Cloud Run
echo "🚀 Deploying to Cloud Run..."
gcloud run deploy ${SERVICE_NAME} \
  --image ${IMAGE_NAME} \
  --platform managed \
  --region ${REGION} \
  --min-instances 1 \
  --no-allow-unauthenticated \
  --set-env-vars=openai_api_key=${OPENAI_API_KEY}
# Step 3: Get the service URL
echo "🔍 Getting service URL..."
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} --region ${REGION} --format="value(status.url)")

echo "========================================"
echo "✅ Deployment complete!"
echo "Service URL: ${SERVICE_URL}"
echo "========================================"