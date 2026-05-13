# Get the directory of this script and navigate to repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEPS_DIR="$REPO_ROOT/deps"

# Create deps directory if it doesn't exist
mkdir -p "$DEPS_DIR"

# Git clone lerobot into deps directory
git clone https://github.com/huggingface/lerobot.git "$DEPS_DIR/lerobot"
git -C "$DEPS_DIR/lerobot" checkout v0.5.1

# Patch: remove Python 3.12-incompatible import in groot/__init__.py
# groot/modeling_groot.py uses a @dataclass with non-default arg after default arg,
# which crashes on Python 3.12. We don't use GrootPolicy, so we simply drop the import.
sed -i '/from .modeling_groot import GrootPolicy/d' \
    "$DEPS_DIR/lerobot/src/lerobot/policies/groot/__init__.py"

# Install lerobot
python -m pip install -e "$DEPS_DIR/lerobot" --no-deps

# Install smolvla extras (transformers, accelerate, safetensors, num2words)
python -m pip install -e "$DEPS_DIR/lerobot[smolvla]" --no-deps
python -m pip install "transformers==5.3.0" "num2words>=0.5.14,<0.6.0" "accelerate>=1.7.0,<2.0.0" "safetensors>=0.4.3,<1.0.0"

# Install a couple of dependencies
python -m pip install -r resfit/lerobot/lerobot_requirements.txt
python -m pip install imageio imageio-ffmpeg
python -m pip install torch==2.11.0+cu128 torchvision torchcodec --index-url https://download.pytorch.org/whl/cu128
python -m pip install "datasets>=4.0.0,<5.0.0"