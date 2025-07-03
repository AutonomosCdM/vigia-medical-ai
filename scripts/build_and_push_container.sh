#!/bin/bash
# VIGIA Medical AI - Container Build and Push Script
# Optimized for AWS ECR deployment with error handling and retry logic

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# =============================================================================
# Configuration
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# AWS Configuration
AWS_REGION="${AWS_REGION:-us-east-1}"
AWS_ACCOUNT_ID="${AWS_ACCOUNT_ID:-$(aws sts get-caller-identity --query Account --output text)}"
ECR_REPOSITORY_NAME="vigia-fastapi"
ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPOSITORY_NAME}"

# Build Configuration
DOCKERFILE="${DOCKERFILE:-Dockerfile.optimized}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
BUILD_TARGET="${BUILD_TARGET:-production}"
PLATFORM="linux/amd64"

# Retry Configuration
MAX_RETRIES=3
RETRY_DELAY=30

# =============================================================================
# Utility Functions
# =============================================================================
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" >&2
}

error() {
    log "ERROR: $*"
    exit 1
}

retry() {
    local max_attempts=$1
    local delay=$2
    shift 2
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        log "Attempt $attempt/$max_attempts: $*"
        if "$@"; then
            return 0
        else
            if [ $attempt -eq $max_attempts ]; then
                error "Command failed after $max_attempts attempts"
            fi
            log "Command failed, retrying in ${delay}s..."
            sleep $delay
            ((attempt++))
        fi
    done
}

check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check AWS CLI
    if ! command -v aws &> /dev/null; then
        error "AWS CLI is not installed"
    fi
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed"
    fi
    
    # Check AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        error "AWS credentials not configured"
    fi
    
    # Check Docker daemon
    if ! docker info &> /dev/null; then
        error "Docker daemon is not running"
    fi
    
    log "Prerequisites check passed"
}

create_ecr_repository() {
    log "Ensuring ECR repository exists..."
    
    # Check if repository exists
    if aws ecr describe-repositories --repository-names "${ECR_REPOSITORY_NAME}" --region "${AWS_REGION}" &> /dev/null; then
        log "ECR repository ${ECR_REPOSITORY_NAME} already exists"
    else
        log "Creating ECR repository ${ECR_REPOSITORY_NAME}..."
        aws ecr create-repository \
            --repository-name "${ECR_REPOSITORY_NAME}" \
            --region "${AWS_REGION}" \
            --image-scanning-configuration scanOnPush=true \
            --encryption-configuration encryptionType=AES256
        log "ECR repository created successfully"
    fi
}

docker_login() {
    log "Logging in to ECR..."
    retry $MAX_RETRIES 10 aws ecr get-login-password --region "${AWS_REGION}" | \
        docker login --username AWS --password-stdin "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
    log "ECR login successful"
}

build_image() {
    log "Building Docker image..."
    log "Using Dockerfile: ${DOCKERFILE}"
    log "Target: ${BUILD_TARGET}"
    log "Platform: ${PLATFORM}"
    
    # Clean up any existing build cache for this image
    docker system prune -f --filter "label=vigia-build=true" &> /dev/null || true
    
    # Build with optimized settings
    docker build \
        --file "${DOCKERFILE}" \
        --target "${BUILD_TARGET}" \
        --platform "${PLATFORM}" \
        --tag "${ECR_REPOSITORY_NAME}:${IMAGE_TAG}" \
        --tag "${ECR_REPOSITORY_NAME}:$(date +%Y%m%d-%H%M%S)" \
        --label "vigia-build=true" \
        --label "build-date=$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
        --label "git-commit=$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')" \
        --progress=plain \
        .
    
    log "Docker image built successfully"
}

tag_image() {
    log "Tagging image for ECR..."
    docker tag "${ECR_REPOSITORY_NAME}:${IMAGE_TAG}" "${ECR_URI}:${IMAGE_TAG}"
    docker tag "${ECR_REPOSITORY_NAME}:${IMAGE_TAG}" "${ECR_URI}:latest"
    log "Image tagged successfully"
}

push_image() {
    log "Pushing image to ECR..."
    
    # Push with retry logic and progress monitoring
    push_with_retry() {
        # Push latest tag
        docker push "${ECR_URI}:latest" && \
        # Push specific tag if different from latest
        if [ "${IMAGE_TAG}" != "latest" ]; then
            docker push "${ECR_URI}:${IMAGE_TAG}"
        fi
    }
    
    retry $MAX_RETRIES $RETRY_DELAY push_with_retry
    log "Image pushed successfully"
}

verify_deployment() {
    log "Verifying deployment..."
    
    # Check if image exists in ECR
    if aws ecr describe-images \
        --repository-name "${ECR_REPOSITORY_NAME}" \
        --image-ids imageTag="${IMAGE_TAG}" \
        --region "${AWS_REGION}" &> /dev/null; then
        log "Image verification successful"
        
        # Get image details
        aws ecr describe-images \
            --repository-name "${ECR_REPOSITORY_NAME}" \
            --image-ids imageTag="${IMAGE_TAG}" \
            --region "${AWS_REGION}" \
            --query 'imageDetails[0].[imageSizeInBytes,imagePushedAt]' \
            --output table
    else
        error "Image verification failed - image not found in ECR"
    fi
}

cleanup() {
    log "Cleaning up local images..."
    
    # Remove local images to save space
    docker rmi "${ECR_REPOSITORY_NAME}:${IMAGE_TAG}" &> /dev/null || true
    docker rmi "${ECR_URI}:${IMAGE_TAG}" &> /dev/null || true
    docker rmi "${ECR_URI}:latest" &> /dev/null || true
    
    # Clean up build cache
    docker system prune -f --filter "label=vigia-build=true" &> /dev/null || true
    
    log "Cleanup completed"
}

# =============================================================================
# Main Execution
# =============================================================================
main() {
    log "Starting VIGIA Medical AI container build and push"
    log "Configuration:"
    log "  AWS Region: ${AWS_REGION}"
    log "  AWS Account: ${AWS_ACCOUNT_ID}"
    log "  ECR Repository: ${ECR_REPOSITORY_NAME}"
    log "  Image Tag: ${IMAGE_TAG}"
    log "  Build Target: ${BUILD_TARGET}"
    
    check_prerequisites
    create_ecr_repository
    docker_login
    build_image
    tag_image
    push_image
    verify_deployment
    
    log "Container build and push completed successfully!"
    log "Image URI: ${ECR_URI}:${IMAGE_TAG}"
    
    # Optional cleanup (comment out if you want to keep local images)
    cleanup
}

# =============================================================================
# Script Execution
# =============================================================================
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --tag)
                IMAGE_TAG="$2"
                shift 2
                ;;
            --target)
                BUILD_TARGET="$2"
                shift 2
                ;;
            --dockerfile)
                DOCKERFILE="$2"
                shift 2
                ;;
            --no-cleanup)
                cleanup() { log "Skipping cleanup"; }
                shift
                ;;
            --help)
                echo "Usage: $0 [OPTIONS]"
                echo "Options:"
                echo "  --tag TAG           Image tag (default: latest)"
                echo "  --target TARGET     Build target (default: production)"
                echo "  --dockerfile FILE   Dockerfile to use (default: Dockerfile.optimized)"
                echo "  --no-cleanup        Skip cleanup of local images"
                echo "  --help              Show this help"
                exit 0
                ;;
            *)
                error "Unknown option: $1"
                ;;
        esac
    done
    
    main
fi