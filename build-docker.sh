#!/usr/bin/env bash
# Build the app image and push it to Docker Hub.
# Usage: ./build-docker.sh [tag]   (default tag: current git short sha)
set -euo pipefail
cd "$(dirname "$0")"

TAG="${1:-$(git rev-parse --short HEAD)}"
IMAGE="tuiteraz/whisper-webui:${TAG}"

docker build -t "$IMAGE" .
docker push "$IMAGE"
echo "Published $IMAGE"
