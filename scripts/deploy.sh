#!/bin/bash

# Crypto Trading Bot Deployment Script
# Supports Docker Compose and Kubernetes deployments

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="crypto-trading-bot"
DOCKER_IMAGE="crypto-trading-bot:latest"
NAMESPACE="crypto-trading"

# Functions
print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

check_dependencies() {
    print_header "Checking Dependencies"
    
    # Check Docker
    if command -v docker &> /dev/null; then
        print_success "Docker is installed"
    else
        print_error "Docker is not installed"
        exit 1
    fi
    
    # Check Docker Compose
    if command -v docker-compose &> /dev/null; then
        print_success "Docker Compose is installed"
    else
        print_error "Docker Compose is not installed"
        exit 1
    fi
    
    # Check kubectl (optional for Kubernetes deployment)
    if command -v kubectl &> /dev/null; then
        print_success "kubectl is installed"
    else
        print_warning "kubectl is not installed (required for Kubernetes deployment)"
    fi
}

build_docker_image() {
    print_header "Building Docker Image"
    
    cd "$(dirname "$0")/.."
    
    if docker build -t $DOCKER_IMAGE -f docker/Dockerfile .; then
        print_success "Docker image built successfully"
    else
        print_error "Failed to build Docker image"
        exit 1
    fi
}

deploy_docker_compose() {
    print_header "Deploying with Docker Compose"
    
    cd "$(dirname "$0")/../docker"
    
    # Create necessary directories
    mkdir -p ../data ../logs ../backups
    
    # Start services
    if docker-compose up -d; then
        print_success "Docker Compose deployment started"
        echo -e "${BLUE}Services:${NC}"
        echo "  - Trading Bot: http://localhost:8080"
        echo "  - Grafana Dashboard: http://localhost:3000"
        echo "  - Prometheus: http://localhost:9090"
        echo "  - PostgreSQL: localhost:5432"
        echo "  - Redis: localhost:6379"
    else
        print_error "Failed to start Docker Compose services"
        exit 1
    fi
}

deploy_kubernetes() {
    print_header "Deploying to Kubernetes"
    
    cd "$(dirname "$0")/../k8s"
    
    # Check if kubectl is available
    if ! command -v kubectl &> /dev/null; then
        print_error "kubectl is required for Kubernetes deployment"
        exit 1
    fi
    
    # Create namespace
    kubectl apply -f namespace.yaml
    
    # Create persistent volumes
    kubectl apply -f persistent-volumes.yaml
    
    # Create secrets (update with actual values)
    print_warning "Please update k8s/secrets.yaml with your actual API keys before deployment"
    kubectl apply -f secrets.yaml
    
    # Create configmap
    kubectl apply -f configmap.yaml
    
    # Deploy services
    kubectl apply -f services.yaml
    
    # Deploy trading bot
    kubectl apply -f deployment.yaml
    
    print_success "Kubernetes deployment completed"
    echo -e "${BLUE}To check status:${NC}"
    echo "  kubectl get pods -n $NAMESPACE"
    echo "  kubectl get services -n $NAMESPACE"
}

stop_docker_compose() {
    print_header "Stopping Docker Compose Services"
    
    cd "$(dirname "$0")/../docker"
    
    if docker-compose down; then
        print_success "Docker Compose services stopped"
    else
        print_error "Failed to stop Docker Compose services"
    fi
}

stop_kubernetes() {
    print_header "Stopping Kubernetes Deployment"
    
    cd "$(dirname "$0")/../k8s"
    
    kubectl delete -f deployment.yaml
    kubectl delete -f services.yaml
    kubectl delete -f configmap.yaml
    kubectl delete -f secrets.yaml
    kubectl delete -f persistent-volumes.yaml
    kubectl delete -f namespace.yaml
    
    print_success "Kubernetes deployment stopped"
}

show_logs() {
    print_header "Showing Logs"
    
    if [ "$1" = "docker" ]; then
        cd "$(dirname "$0")/../docker"
        docker-compose logs -f trading-bot
    elif [ "$1" = "k8s" ]; then
        kubectl logs -f deployment/trading-bot -n $NAMESPACE
    else
        print_error "Please specify 'docker' or 'k8s' for logs"
    fi
}

show_status() {
    print_header "Service Status"
    
    if [ "$1" = "docker" ]; then
        cd "$(dirname "$0")/../docker"
        docker-compose ps
    elif [ "$1" = "k8s" ]; then
        kubectl get pods -n $NAMESPACE
        kubectl get services -n $NAMESPACE
    else
        print_error "Please specify 'docker' or 'k8s' for status"
    fi
}

# Main script
case "$1" in
    "build")
        check_dependencies
        build_docker_image
        ;;
    "deploy-docker")
        check_dependencies
        build_docker_image
        deploy_docker_compose
        ;;
    "deploy-k8s")
        check_dependencies
        build_docker_image
        deploy_kubernetes
        ;;
    "stop-docker")
        stop_docker_compose
        ;;
    "stop-k8s")
        stop_kubernetes
        ;;
    "logs")
        show_logs "$2"
        ;;
    "status")
        show_status "$2"
        ;;
    "health")
        print_header "Health Check"
        curl -f http://localhost:8080/health || print_error "Health check failed"
        ;;
    *)
        echo "Usage: $0 {build|deploy-docker|deploy-k8s|stop-docker|stop-k8s|logs|status|health}"
        echo ""
        echo "Commands:"
        echo "  build           - Build Docker image"
        echo "  deploy-docker   - Deploy using Docker Compose"
        echo "  deploy-k8s      - Deploy to Kubernetes"
        echo "  stop-docker     - Stop Docker Compose services"
        echo "  stop-k8s        - Stop Kubernetes deployment"
        echo "  logs [docker|k8s] - Show logs"
        echo "  status [docker|k8s] - Show service status"
        echo "  health          - Check health endpoint"
        exit 1
        ;;
esac 