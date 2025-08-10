#!/bin/bash

# Production Deployment Script for Retro-PEFT-Adapters
# Usage: ./deploy.sh [environment] [action]
# Examples:
#   ./deploy.sh production docker
#   ./deploy.sh staging kubernetes
#   ./deploy.sh local test

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEPLOYMENT_DIR="$PROJECT_ROOT/deployment"

# Default values
ENVIRONMENT=${1:-production}
ACTION=${2:-docker}
VERSION=${3:-latest}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if required commands exist
    local required_commands=("docker" "python3" "pip")
    
    if [[ "$ACTION" == "kubernetes" ]]; then
        required_commands+=("kubectl" "helm")
    fi
    
    for cmd in "${required_commands[@]}"; do
        if ! command -v "$cmd" &> /dev/null; then
            log_error "Required command '$cmd' not found"
            exit 1
        fi
    done
    
    log_success "Prerequisites check passed"
}

# Build Docker image
build_docker_image() {
    log_info "Building Docker image for $ENVIRONMENT environment..."
    
    cd "$PROJECT_ROOT"
    
    # Build image
    docker build \
        --file "$DEPLOYMENT_DIR/docker/Dockerfile" \
        --target production \
        --tag "retro-peft-adapters:$VERSION" \
        --tag "retro-peft-adapters:latest" \
        .
    
    log_success "Docker image built successfully"
}

# Run tests
run_tests() {
    log_info "Running test suite..."
    
    cd "$PROJECT_ROOT"
    
    # Create test environment
    python3 -m venv test_env
    source test_env/bin/activate
    
    # Install dependencies
    pip install --upgrade pip
    pip install -e ".[test]"
    pip install pytest pytest-asyncio pytest-cov
    
    # Run tests
    PYTHONPATH="$PROJECT_ROOT/src" python -m pytest tests/ -v --tb=short
    
    # Cleanup
    deactivate
    rm -rf test_env
    
    log_success "Tests passed successfully"
}

# Deploy with Docker Compose
deploy_docker() {
    log_info "Deploying with Docker Compose..."
    
    cd "$DEPLOYMENT_DIR/docker"
    
    # Create required directories
    mkdir -p logs data models
    
    # Pull/build images
    docker-compose pull || true
    docker-compose build
    
    # Deploy services
    docker-compose up -d
    
    # Wait for services to be healthy
    log_info "Waiting for services to be healthy..."
    sleep 10
    
    # Check service health
    if docker-compose ps | grep -q "Up (healthy)"; then
        log_success "Services are healthy"
    else
        log_warning "Some services may not be fully healthy yet"
        docker-compose ps
    fi
    
    log_success "Docker deployment completed"
    log_info "API available at: http://localhost:8000"
    log_info "Grafana dashboard: http://localhost:3000 (admin/admin123)"
    log_info "Prometheus: http://localhost:9090"
}

# Deploy to Kubernetes
deploy_kubernetes() {
    log_info "Deploying to Kubernetes..."
    
    # Check if kubectl is configured
    if ! kubectl cluster-info &> /dev/null; then
        log_error "kubectl is not configured or cluster is not accessible"
        exit 1
    fi
    
    # Apply namespace and RBAC
    kubectl apply -f "$DEPLOYMENT_DIR/kubernetes/retro-peft-deployment.yaml"
    
    # Wait for deployment
    log_info "Waiting for deployment to be ready..."
    kubectl wait --for=condition=available --timeout=300s deployment/retro-peft-api -n retro-peft
    
    # Get service information
    kubectl get services -n retro-peft
    
    log_success "Kubernetes deployment completed"
    
    # Get ingress information if available
    if kubectl get ingress retro-peft-ingress -n retro-peft &> /dev/null; then
        log_info "Ingress configuration:"
        kubectl get ingress retro-peft-ingress -n retro-peft
    fi
}

# Health check
health_check() {
    log_info "Performing health check..."
    
    local health_url
    
    if [[ "$ACTION" == "docker" ]]; then
        health_url="http://localhost:8000/health"
    elif [[ "$ACTION" == "kubernetes" ]]; then
        # Port-forward for health check
        kubectl port-forward service/retro-peft-api-service 8080:8000 -n retro-peft &
        local port_forward_pid=$!
        sleep 5
        health_url="http://localhost:8080/health"
    else
        log_warning "Health check not supported for action: $ACTION"
        return
    fi
    
    # Perform health check
    if curl -f -s "$health_url" > /dev/null; then
        log_success "Health check passed"
    else
        log_error "Health check failed"
        exit 1
    fi
    
    # Cleanup port-forward if used
    if [[ -n "${port_forward_pid:-}" ]]; then
        kill "$port_forward_pid" 2>/dev/null || true
    fi
}

# Cleanup function
cleanup() {
    log_info "Cleaning up..."
    
    if [[ "$ACTION" == "docker" ]]; then
        cd "$DEPLOYMENT_DIR/docker"
        docker-compose down -v
        docker system prune -f
    elif [[ "$ACTION" == "kubernetes" ]]; then
        kubectl delete namespace retro-peft --ignore-not-found=true
    fi
    
    log_success "Cleanup completed"
}

# Show usage
show_usage() {
    echo "Usage: $0 [environment] [action] [version]"
    echo ""
    echo "Environments:"
    echo "  production  - Production deployment (default)"
    echo "  staging     - Staging deployment"
    echo "  local       - Local development"
    echo ""
    echo "Actions:"
    echo "  docker      - Deploy with Docker Compose (default)"
    echo "  kubernetes  - Deploy to Kubernetes"
    echo "  test        - Run test suite only"
    echo "  build       - Build Docker image only"
    echo "  cleanup     - Remove deployment"
    echo ""
    echo "Examples:"
    echo "  $0 production docker latest"
    echo "  $0 staging kubernetes v1.2.0"
    echo "  $0 local test"
    echo "  $0 production cleanup"
}

# Main execution
main() {
    log_info "Starting deployment for environment: $ENVIRONMENT, action: $ACTION"
    
    case "$ACTION" in
        "test")
            check_prerequisites
            run_tests
            ;;
        "build")
            check_prerequisites
            build_docker_image
            ;;
        "docker")
            check_prerequisites
            build_docker_image
            run_tests
            deploy_docker
            health_check
            ;;
        "kubernetes")
            check_prerequisites
            build_docker_image
            run_tests
            deploy_kubernetes
            health_check
            ;;
        "cleanup")
            cleanup
            ;;
        "help"|"-h"|"--help")
            show_usage
            ;;
        *)
            log_error "Unknown action: $ACTION"
            show_usage
            exit 1
            ;;
    esac
    
    log_success "Deployment process completed successfully!"
}

# Handle interrupts
trap 'log_error "Deployment interrupted"; exit 1' INT TERM

# Run main function
main "$@"