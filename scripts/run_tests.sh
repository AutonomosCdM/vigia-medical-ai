#!/bin/bash
# 
# Test execution script for Vigia Medical System
# Standardized test runner with different test suites
#

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to setup test environment
setup_test_env() {
    print_status "Setting up test environment..."
    
    # Check if .env.testing exists
    if [[ ! -f ".env.testing" ]]; then
        print_error ".env.testing file not found!"
        print_status "Create .env.testing with required environment variables"
        exit 1
    fi
    
    # Load testing environment
    export $(grep -v '^#' .env.testing | xargs)
    export TESTING=true
    
    print_success "Test environment configured"
}

# Function to run validation script
run_validation() {
    print_status "Running post-refactor validation..."
    
    if [[ -f "scripts/validate_post_refactor_simple.py" ]]; then
        python scripts/validate_post_refactor_simple.py --verbose
    else
        print_warning "Validation script not found, skipping..."
    fi
}

# Function to run specific test suite
run_test_suite() {
    local suite="$1"
    local description="$2"
    local path="$3"
    local markers="${4:-}"
    
    print_status "Running $description..."
    
    if [[ ! -d "$path" && ! -f "$path" ]]; then
        print_warning "$description tests not found at $path, skipping..."
        return 0
    fi
    
    local cmd="python -m pytest $path -v"
    
    if [[ -n "$markers" ]]; then
        cmd="$cmd -m \"$markers\""
    fi
    
    if eval "$cmd"; then
        print_success "$description completed successfully"
        return 0
    else
        print_error "$description failed"
        return 1
    fi
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Test execution options:"
    echo "  all         Run all tests (default)"
    echo "  unit        Run unit tests only"
    echo "  integration Run integration tests only"
    echo "  e2e         Run end-to-end tests only"
    echo "  smoke       Run smoke tests only"
    echo "  medical     Run medical/clinical tests only"
    echo "  security    Run security tests only"
    echo "  critical    Run critical deployment tests only"
    echo "  validate    Run validation script only"
    echo "  coverage    Run tests with coverage report"
    echo "  quick       Run quick validation (smoke + unit)"
    echo ""
    echo "Options:"
    echo "  -h, --help  Show this help message"
    echo "  --no-setup  Skip test environment setup"
    echo "  --verbose   Enable verbose output"
    echo ""
    echo "Examples:"
    echo "  $0                    # Run all tests"
    echo "  $0 e2e               # Run only E2E tests"
    echo "  $0 quick             # Run quick validation"
    echo "  $0 coverage          # Run with coverage"
}

# Parse command line arguments
SUITE="all"
SKIP_SETUP=false
VERBOSE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_usage
            exit 0
            ;;
        --no-setup)
            SKIP_SETUP=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        all|unit|integration|e2e|smoke|medical|security|critical|validate|coverage|quick)
            SUITE="$1"
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Main execution
main() {
    print_status "🩺 Vigia Medical System - Test Runner"
    print_status "====================================="
    
    # Check dependencies
    if ! command_exists python; then
        print_error "Python not found!"
        exit 1
    fi
    
    if ! command_exists pytest; then
        print_error "pytest not found! Install with: pip install pytest"
        exit 1
    fi
    
    # Setup test environment
    if [[ "$SKIP_SETUP" == false ]]; then
        setup_test_env
    fi
    
    # Track test results
    FAILED_SUITES=()
    
    case "$SUITE" in
        "validate")
            run_validation
            ;;
        "unit")
            run_test_suite "unit" "Unit Tests" "tests" "unit" || FAILED_SUITES+=("unit")
            ;;
        "integration")
            run_test_suite "integration" "Integration Tests" "tests" "integration" || FAILED_SUITES+=("integration")
            ;;
        "e2e")
            run_test_suite "e2e" "End-to-End Tests" "tests/e2e" "e2e" || FAILED_SUITES+=("e2e")
            ;;
        "smoke")
            run_test_suite "smoke" "Smoke Tests" "tests" "smoke" || FAILED_SUITES+=("smoke")
            ;;
        "medical")
            run_test_suite "medical" "Medical/Clinical Tests" "tests" "medical" || FAILED_SUITES+=("medical")
            ;;
        "security")
            run_test_suite "security" "Security Tests" "tests/security" "security" || FAILED_SUITES+=("security")
            ;;
        "critical")
            run_test_suite "critical" "Critical Deployment Tests" "tests" "critical" || FAILED_SUITES+=("critical")
            ;;
        "coverage")
            print_status "Running tests with coverage..."
            python -m pytest tests/ --cov=src --cov-report=html --cov-report=term-missing -v || FAILED_SUITES+=("coverage")
            print_status "Coverage report generated in htmlcov/"
            ;;
        "quick")
            print_status "Running quick validation (smoke + basic tests)..."
            run_test_suite "smoke" "Smoke Tests" "tests" "smoke" || FAILED_SUITES+=("smoke")
            run_test_suite "basic" "Basic Integration Tests" "tests/e2e/test_simple_integration.py" || FAILED_SUITES+=("basic")
            run_validation
            ;;
        "all")
            print_status "Running complete test suite..."
            
            # Run validation first
            run_validation
            
            # Run all test suites
            run_test_suite "unit" "Unit Tests" "tests" "unit" || FAILED_SUITES+=("unit")
            run_test_suite "integration" "Integration Tests" "tests" "integration" || FAILED_SUITES+=("integration") 
            run_test_suite "e2e" "End-to-End Tests" "tests/e2e" || FAILED_SUITES+=("e2e")
            run_test_suite "security" "Security Tests" "tests/security" || FAILED_SUITES+=("security")
            ;;
    esac
    
    # Summary
    echo ""
    print_status "====================================="
    print_status "🩺 Test Execution Summary"
    print_status "====================================="
    
    if [[ ${#FAILED_SUITES[@]} -eq 0 ]]; then
        print_success "✅ All tests completed successfully!"
        print_status "System is ready for deployment"
        exit 0
    else
        print_error "❌ Some test suites failed:"
        for suite in "${FAILED_SUITES[@]}"; do
            print_error "  • $suite"
        done
        print_status "Review failed tests before deployment"
        exit 1
    fi
}

# Run main function
main "$@"