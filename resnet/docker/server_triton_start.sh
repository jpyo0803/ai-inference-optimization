#!/bin/bash

echo "Starting Triton Inference Server..."
tritonserver --model-repository=/models \
             --http-port=8080 \
             --grpc-port=8001 \
             --metrics-port=8002 &

echo "Waiting for Triton to be ready..."
while ! curl -s localhost:8080/v2/health/ready > /dev/null; do
    sleep 1
done
echo "Triton Server is Ready!"

echo "Starting FastAPI Server for model: $MODEL_NAME"
python3 server_triton.py