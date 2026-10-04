#!/usr/bin/env bash
# Build and publish the whisper-webui image to Docker Hub.
#
# Tags built+pushed: tuiteraz/whisper-webui:<tag> and tuiteraz/whisper-webui:latest
#   <tag> defaults to the current git short sha; override with the first argument:
#     ./build.sh v1.2.3
#
# Platform: defaults to linux/amd64 (the deployment target, e.g. puma.lan).
# Override for other targets: DOCKER_PLATFORMS="linux/arm64" ./build.sh
set -euo pipefail

IMAGE_REPO="tuiteraz/whisper-webui"
PLATFORM="${DOCKER_PLATFORMS:-linux/amd64}"

cd "$(dirname "$0")"

if [[ $# -ge 1 && -n "$1" ]]; then
  TAG="$1"
else
  TAG="$(git rev-parse --short HEAD)"
fi

echo "==> Building ${IMAGE_REPO}:${TAG} + :latest (platform: ${PLATFORM})"
if ! docker buildx build \
  --platform "${PLATFORM}" \
  -t "${IMAGE_REPO}:${TAG}" \
  -t "${IMAGE_REPO}:latest" \
  --push \
  .; then
  echo "ERROR: docker buildx build --push failed for ${IMAGE_REPO}:${TAG}." >&2
  echo "       If the build itself succeeded, this is a Docker Hub auth/push failure." >&2
  echo "       Log in first:  docker login -u <dockerhub-username>" >&2
  echo "   or:  echo \"\$DOCKERHUB_TOKEN\" | docker login -u \"\$DOCKERHUB_USERNAME\" --password-stdin" >&2
  exit 1
fi

echo "==> Published ${IMAGE_REPO}:${TAG} and ${IMAGE_REPO}:latest"
