# Your custom Dockerfile for in-orbit inference
FROM tm2space/aicube-base:latest

# Only add packages NOT in base image (check library docs first!)
# The base image already includes: PyTorch, TensorFlow, GDAL, Rasterio, etc.
COPY requirements.txt /tmp/
RUN pip3 install --no-cache-dir --index-url https://pypi.org/simple \
    --extra-index-url https://pypi.ngc.nvidia.com \
    -r /tmp/requirements.txt && \
    rm -rf ~/.cache/pip

# Copy source files directly to /workspace/
COPY ./aicube_image_loader.py /workspace/
COPY ./crop_health_app.py /workspace/
COPY ./eurosat_resnet18.onnx /workspace/

# Models and config at workspace level

WORKDIR /workspace

# Entry point for your application
CMD ["python3", "crop_health_app.py"]