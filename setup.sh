#!/bin/bash
# Setup script for Video Editor with SAM 2
set -e

cd "$(dirname "$0")"

echo "=== Video Editor Setup ==="

# Create venv
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

# Install deps
echo "Installing dependencies..."
pip install --upgrade pip
pip install torch torchvision opencv-python numpy pillow

# Install SAM 2
echo "Installing SAM 2..."
pip install sam-2

# Download checkpoint
CKPT_DIR="checkpoints"
CKPT_FILE="$CKPT_DIR/sam2.1_hiera_large.pt"
if [ ! -f "$CKPT_FILE" ]; then
    echo "Downloading SAM 2.1 Hiera Large checkpoint (~2.4GB)..."
    mkdir -p "$CKPT_DIR"
    curl -L -o "$CKPT_FILE" \
        "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
    echo "Checkpoint downloaded."
else
    echo "Checkpoint already exists: $CKPT_FILE"
fi

# Check ffmpeg
if command -v ffmpeg &> /dev/null; then
    echo "ffmpeg: $(ffmpeg -version | head -1)"
else
    echo "WARNING: ffmpeg not found. Audio won't be copied to output."
    echo "Install with: brew install ffmpeg"
fi

echo ""
echo "=== Setup complete ==="
echo ""
echo "Usage:"
echo "  source venv/bin/activate"
echo "  python run_pipeline.py --video inputs/video.mp4 --background inputs/bg.jpg"
