#!/bin/bash
# Build and push the reward hacking environment Docker image
#
# Usage:
#   ./build_and_push.sh --registry myorg    # Build only
#   ./build_and_push.sh --registry myorg --push  # Build and push to registry
#
# The image name will be: <registry>/reward-hacking-env:latest

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REGISTRY="${REGISTRY:-}"
IMAGE_NAME="reward-hacking-env"
TAG="${TAG:-latest}"
PUSH=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --push)
            PUSH=true
            shift
            ;;
        --registry)
            REGISTRY="$2"
            shift 2
            ;;
        --tag)
            TAG="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [ -z "$REGISTRY" ]; then
    echo "Error: --registry is required (e.g. --registry myorg)"
    exit 1
fi

FULL_IMAGE="${REGISTRY}/${IMAGE_NAME}:${TAG}"

echo "Building Docker image: ${FULL_IMAGE}"
docker build -t "${FULL_IMAGE}" "${SCRIPT_DIR}"

if [ "$PUSH" = true ]; then
    echo "Pushing image to registry..."
    docker push "${FULL_IMAGE}"
    echo "Image pushed: ${FULL_IMAGE}"
    echo ""
    echo "Update k8s_values.yaml with:"
    echo "  image: ${FULL_IMAGE}"
else
    echo ""
    echo "Image built locally: ${FULL_IMAGE}"
    echo "To push, run: $0 --push"
fi
