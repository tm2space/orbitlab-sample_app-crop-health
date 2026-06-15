#!/bin/bash
# AICube Uplink Package Creation Tool
# Creates optimized uplink packages for satellite upload

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_IMAGE="tm2space/aicube-base:latest"
MAX_UPLINK_SIZE_MB=200
TEMP_DIR="$HOME/.cache/aicube_uplink_$$"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step() { echo -e "${BLUE}[STEP]${NC} $1"; }

# Function to show usage
show_usage() {
    cat << EOF
AICube Uplink Package Creation Tool

Usage: $0 <customer_image> [output_file] [task_id]

Arguments:
  customer_image  Your Docker image (e.g., my-app:latest)
  output_file     Output uplink package (default: customer_image_uplink.tar.gz)
  task_id         Task identifier for tracking (optional)

Examples:
  $0 my-gis-app:latest
  $0 change-detection:v1.2 my_uplink.tar.gz 12345
  $0 crop-monitor:latest crop_monitor_uplink.tar.gz

Options:
  --validate-only    Only validate, don't create uplink package
  --max-size SIZE    Maximum uplink size in MB (default: 200)
  --help            Show this help

The tool will:
1. Analyze your image vs base image
2. Extract only the differences (uplink package)
3. Compress and optimize for satellite upload
4. Validate size limits and structure
5. Create upload-ready package

EOF
}

# Function to validate base image exists
check_base_image() {
    if ! docker images --format "{{.Repository}}:{{.Tag}}" | grep -q "^${BASE_IMAGE}$"; then
        log_error "Base image not found: $BASE_IMAGE"
        log_error "Please pull the base image first:"
        log_error "  docker pull $BASE_IMAGE"
        exit 1
    fi
    log_info "Base image found: $BASE_IMAGE"
}

# Function to validate customer image
check_customer_image() {
    local customer_image=$1

    if ! docker images --format "{{.Repository}}:{{.Tag}}" | grep -q "^${customer_image}$"; then
        log_error "Customer image not found: $customer_image"
        log_error "Please build your image first:"
        log_error "  docker build -t $customer_image ."
        exit 1
    fi

    # Check if image is based on our base image
    local base_layers=$(docker inspect $BASE_IMAGE --format='{{range .RootFS.Layers}}{{.}} {{end}}' | tr ' ' '\n' | sort)
    local customer_layers=$(docker inspect $customer_image --format='{{range .RootFS.Layers}}{{.}} {{end}}' | tr ' ' '\n' | sort)

    local base_in_customer=$(comm -12 <(echo "$base_layers") <(echo "$customer_layers") | wc -l)
    local total_base=$(echo "$base_layers" | wc -l)

    if [ $base_in_customer -lt $((total_base * 80 / 100)) ]; then
        log_warn "Customer image doesn't appear to be based on $BASE_IMAGE"
        log_warn "This may result in a larger delta than necessary"
        echo -n "Continue anyway? (y/N): "
        read response
        if [[ "$response" != "y" ]]; then
            exit 1
        fi
    fi

    log_info "Customer image validated: $customer_image"
}

# Function to analyze image differences
analyze_differences() {
    local customer_image=$1

    log_step "Analyzing differences between base and customer images"

    # Get image sizes
    local base_size=$(docker images $BASE_IMAGE --format "{{.Size}}" | head -1)
    local customer_size=$(docker images $customer_image --format "{{.Size}}" | head -1)

    log_info "Base image size: $base_size"
    log_info "Customer image size: $customer_size"

    # Show layer differences
    echo ""
    log_info "Layer differences:"
    docker history $customer_image --no-trunc --format "table {{.CreatedBy}}\t{{.Size}}" | head -10
}

# Function to create uplink package
create_uplink_package() {
    local customer_image=$1
    local output_file=$2
    local task_id=${3:-$(date +%s)}

    log_step "Creating uplink package for $customer_image"

    # Convert output_file to absolute path before changing directories
    output_file=$(realpath "$output_file")

    # Create temporary directory
    mkdir -p $TEMP_DIR
    mkdir -p $TEMP_DIR/delta_layers

    log_info "Extracting delta layers (differences from base image)..."

    # Get base image layers
    local base_layers=$(docker inspect $BASE_IMAGE --format='{{range .RootFS.Layers}}{{.}} {{end}}')

    # Get customer image layers
    local customer_layers=$(docker inspect $customer_image --format='{{range .RootFS.Layers}}{{.}} {{end}}')

    # Find delta layers (layers in customer but not in base)
    local delta_count=0
    for layer in $customer_layers; do
        if ! echo "$base_layers" | grep -q "$layer"; then
            delta_count=$((delta_count + 1))
        fi
    done

    log_info "Found $delta_count delta layer(s)"

    # Save the customer image to extract delta layers
    log_info "Saving customer image..."
    docker save $customer_image -o $TEMP_DIR/customer_image.tar

    # Extract the image tar to analyze layers
    cd $TEMP_DIR
    mkdir -p image_extract
    tar -xf customer_image.tar -C image_extract

    # Get the image config to find layer tar files
    local manifest=$(cat image_extract/manifest.json)
    local layer_files=$(echo "$manifest" | python3 -c "import sys, json; data=json.load(sys.stdin); print(' '.join(data[0]['Layers']))")

    # Copy only delta layers
    log_info "Extracting delta layers..."
    local layer_index=0
    local delta_layer_index=0

    for layer_file in $layer_files; do
        local layer_hash=$(echo "$customer_layers" | cut -d' ' -f$((layer_index + 1)))

        # Check if this layer is a delta (not in base)
        if ! echo "$base_layers" | grep -q "$layer_hash"; then
            log_info "  Delta layer $((delta_layer_index + 1)): ${layer_file}"
            cp "image_extract/$layer_file" "delta_layers/layer_${delta_layer_index}.tar"
            delta_layer_index=$((delta_layer_index + 1))
        fi

        layer_index=$((layer_index + 1))
    done

    # Create uplink package metadata
    cat > uplink_metadata.json << EOF
{
  "format_version": "1.0",
  "base_image": "$BASE_IMAGE",
  "customer_image": "$customer_image",
  "task_id": "$task_id",
  "created_at": "$(date -u +'%Y-%m-%dT%H:%M:%SZ')",
  "created_by": "aicube-delta-tool",
  "extraction_method": "layer_diff",
  "delta_layers": $delta_count,
  "compression": "gzip"
}
EOF

    # Create the final uplink package with only delta layers
    log_info "Creating compressed delta package..."
    tar -czf uplink_package.tar.gz delta_layers/ uplink_metadata.json

    # Move to final location
    mv uplink_package.tar.gz "$output_file"

    # Cleanup
    cd - > /dev/null
    rm -rf $TEMP_DIR

    log_info "Delta package created: $output_file"
}

# Function to validate uplink package
validate_uplink_package() {
    local uplink_file=$1
    local max_size_mb=${2:-$MAX_UPLINK_SIZE_MB}

    log_step "Validating uplink package: $uplink_file"

    if [ ! -f "$uplink_file" ]; then
        log_error "Uplink file not found: $uplink_file"
        return 1
    fi

    # Check file size
    local size_bytes=$(stat -c%s "$uplink_file")
    local size_mb=$((size_bytes / 1024 / 1024))

    log_info "Uplink size: ${size_mb} MB ($(numfmt --to=iec-i $size_bytes))"

    if [ $size_mb -gt $max_size_mb ]; then
        log_error "Uplink size (${size_mb} MB) exceeds maximum (${max_size_mb} MB)"
        log_error "Consider optimizing your container:"
        log_error "  - Remove unnecessary files"
        log_error "  - Use .dockerignore"
        log_error "  - Use multi-stage builds"
        log_error "  - Avoid installing packages already in base image"
        return 1
    else
        log_info "✅ Uplink size OK (${size_mb} MB <= ${max_size_mb} MB)"
    fi

    # Validate package structure
    log_info "Validating package structure..."
    if tar -tzf "$uplink_file" | grep -q "uplink_metadata.json"; then
        log_info "✅ Metadata found"
    else
        log_error "❌ Missing metadata file"
        return 1
    fi

    if tar -tzf "$uplink_file" | grep -q "delta_layers/"; then
        log_info "✅ Delta layers found"
    else
        log_error "❌ Missing delta layers"
        return 1
    fi

    # Extract and validate metadata
    local temp_meta="/tmp/uplink_meta_$.json"
    tar -Ozf "$uplink_file" uplink_metadata.json > "$temp_meta"

    local base_image=$(python3 -c "import json; print(json.load(open('$temp_meta')).get('base_image', ''))")
    if [ "$base_image" == "$BASE_IMAGE" ]; then
        log_info "✅ Base image match: $base_image"
    else
        log_warn "⚠ Base image mismatch: expected $BASE_IMAGE, got $base_image"
    fi

    rm -f "$temp_meta"

    log_info "✅ Uplink package validation successful"
    return 0
}

# Function to estimate upload time
estimate_upload_time() {
    local uplink_file=$1
    local bandwidth_kbps=${2:-128}  # Default satellite uplink

    local size_bytes=$(stat -c%s "$uplink_file")
    local size_mb=$((size_bytes / 1024 / 1024))
    local upload_seconds=$((size_bytes * 8 / bandwidth_kbps / 1000))
    local upload_minutes=$((upload_seconds / 60))

    echo ""
    log_info "Upload estimation:"
    log_info "  File size: ${size_mb} MB"
    log_info "  Satellite uplink: ${bandwidth_kbps} kbps"
    log_info "  Estimated upload time: ${upload_minutes} minutes"

    if [ $upload_minutes -gt 60 ]; then
        log_warn "Upload time > 1 hour - consider further optimization"
    elif [ $upload_minutes -gt 30 ]; then
        log_warn "Upload time > 30 minutes - may want to optimize further"
    else
        log_info "✅ Upload time reasonable for satellite operations"
    fi
}

# Function to show optimization tips
show_optimization_tips() {
    cat << EOF

${YELLOW}=== Uplink Package Optimization Tips ===${NC}

If your uplink package is too large, try these optimization strategies:

${BLUE}1. Multi-stage Docker build:${NC}
FROM tm2space/aicube-base:latest as builder
WORKDIR /build
COPY requirements.txt .
RUN pip3 install --user -r requirements.txt
COPY src/ .

FROM tm2space/aicube-base:latest
COPY --from=builder /root/.local /root/.local
COPY --from=builder /build/app.py /workspace/

${BLUE}2. Use .dockerignore:${NC}
**/__pycache__
*.pyc
*.pyo
.git
.gitignore
README.md
tests/
docs/
*.log

${BLUE}3. Minimize additional packages:${NC}
# Check what's already in base image first!
# Only install packages NOT in tm2space/aicube-base:latest

${BLUE}4. Remove development files:${NC}
RUN pip3 install --no-cache-dir -r requirements.txt && \\
    rm -rf ~/.cache/pip && \\
    find . -name "*.pyc" -delete

${BLUE}5. Compress static files:${NC}
# Pre-compress large config files, models, etc.

EOF
}

# Main execution
main() {
    local customer_image=""
    local output_file=""
    local task_id=""
    local validate_only=false
    local max_size_mb=$MAX_DELTA_SIZE_MB

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --help)
                show_usage
                exit 0
                ;;
            --validate-only)
                validate_only=true
                shift
                ;;
            --max-size)
                max_size_mb="$2"
                shift 2
                ;;
            --*)
                log_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
            *)
                if [ -z "$customer_image" ]; then
                    customer_image="$1"
                elif [ -z "$output_file" ]; then
                    output_file="$1"
                elif [ -z "$task_id" ]; then
                    task_id="$1"
                else
                    log_error "Too many arguments"
                    show_usage
                    exit 1
                fi
                shift
                ;;
        esac
    done

    # Validate required arguments
    if [ -z "$customer_image" ]; then
        log_error "Customer image required"
        show_usage
        exit 1
    fi

    # Set default output file
    if [ -z "$output_file" ]; then
        output_file="${customer_image//[^a-zA-Z0-9._-]/_}_uplink.tar.gz"
    fi

    # Show header
    echo ""
    log_info "AICube Delta Creation Tool"
    log_info "Customer Image: $customer_image"
    log_info "Output File: $output_file"
    log_info "Max Size: ${max_size_mb} MB"
    echo ""

    # Check prerequisites
    check_base_image
    check_customer_image "$customer_image"

    # Analyze differences
    analyze_differences "$customer_image"

    if [ "$validate_only" = true ]; then
        log_info "Validation complete (--validate-only specified)"
        exit 0
    fi

    # Create uplink package
    create_uplink_package "$customer_image" "$output_file" "$task_id"

    # Validate the created package
    if validate_uplink_package "$output_file" "$max_size_mb"; then
        log_info "✅ Delta package ready for satellite upload!"
        estimate_upload_time "$output_file"

        echo ""
        log_info "Next steps:"
        log_info "1. Upload $output_file via OrbitLab Dashboard"
        log_info "2. Configure task parameters"
        log_info "3. Schedule execution time"

    else
        log_error "❌ Delta package validation failed"
        show_optimization_tips
        exit 1
    fi
}

# Cleanup on exit
trap 'rm -rf $TEMP_DIR' EXIT

main "$@"
