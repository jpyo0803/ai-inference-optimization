#!/bin/bash

export HOST_ABS_PATH=$(pwd)/model_analyzer

docker run --rm -it --net=host --gpus all \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v "$HOST_ABS_PATH:$HOST_ABS_PATH" \
  -w "$HOST_ABS_PATH" \
  nvcr.io/nvidia/tritonserver:24.12-py3-sdk \
  model-analyzer profile \
  --model-repository "$HOST_ABS_PATH/models" \
  --output-model-repository-path "$HOST_ABS_PATH/results/output_models" \
  --export-path "$HOST_ABS_PATH/results" \
  --config-file "$HOST_ABS_PATH/profile.yaml" \
  --triton-docker-mounts "$HOST_ABS_PATH:$HOST_ABS_PATH:rw" \
  --override-output-model-repository \
  --triton-launch-mode=docker \
  --triton-docker-image=nvcr.io/nvidia/tritonserver:24.12-py3