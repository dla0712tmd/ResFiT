#!/usr/bin/env bash
# setup_libero_plus.sh — install LIBERO-plus and all dependencies
#
# LIBERO-plus extends the four standard LIBERO task suites with ~10 000
# perturbation variants (layout, cameras, lighting, textures, language, noise).
# The training dataset (lerobot/libero_plus) is auto-downloaded by lerobot.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEPS_DIR="$REPO_ROOT/deps"

mkdir -p "$DEPS_DIR"

# ── 1. System packages (ImageMagick / Wand required by LIBERO-plus) ────────
apt-get update -qq
apt-get install -y --no-install-recommends \
    libexpat1 libfontconfig1-dev libmagickwand-dev

# ── 2. LIBERO-plus (replaces standard LIBERO) ──────────────────────────────
# robosuite==1.4.0 is declared in LIBERO-plus requirements.txt and is
# installed automatically as a transitive dependency.
[ -d "$DEPS_DIR/libero" ] || \
    git clone https://github.com/sylvestf/LIBERO-plus.git "$DEPS_DIR/libero"
python -m pip install -e "$DEPS_DIR/libero"
python -m pip install \
    bddl==1.0.1 easydict==1.9 future==0.18.2 cloudpickle==2.1.0 "gym==0.25.2"

# Workaround for editable-install path-mapping bug
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
echo "$DEPS_DIR/libero" > "$SITE_PACKAGES/libero-path.pth"

# Pre-generate libero config to avoid interactive prompt on first import
echo "N" | python -c "from libero.libero import benchmark" 2>/dev/null || true

# ── 3. LIBERO-plus extra assets (objects, textures, init-states, BDDL files) ─
# assets.zip (~6.4 GB) from Sylvest/LIBERO-plus contains additional objects,
# textures, init-states, and BDDL files needed for the perturbation variants.
# Required for evaluation rollouts; not needed for dataset-only training.
LIBERO_ASSETS_DIR="$DEPS_DIR/libero/libero/libero/assets"
mkdir -p "$LIBERO_ASSETS_DIR"

python - "$LIBERO_ASSETS_DIR" <<'PYEOF'
import pathlib, sys, zipfile
from huggingface_hub import hf_hub_download

assets_dst = pathlib.Path(sys.argv[1])
print(f"Downloading LIBERO-plus assets.zip (~6.4 GB) → {assets_dst}")

zip_path = hf_hub_download(
    repo_id="Sylvest/LIBERO-plus",
    filename="assets.zip",
    repo_type="dataset",
    local_dir="/tmp/libero_plus_assets",
)
print(f"Extracting {zip_path} …")
with zipfile.ZipFile(zip_path) as z:
    z.extractall(assets_dst)
print("Assets ready.")
PYEOF

# ── 4. Python dependencies (single consolidated install) ───────────────────
python -m pip install \
    gymnasium==1.1.1 \
    mujoco==3.3.2 \
    PyOpenGL-accelerate \
    diffusers==0.33.1 \
    "protobuf>4.21.0,<5" \
    multidict==6.0.5 \
    torchrl==0.8.0 tensordict==0.8.2 torchcodec==0.11.1 \
    "numba>=0.60" "llvmlite>=0.44" \
    tabulate ipdb pyserial deepdiff matplotlib draccus==0.10.0

# ── 5. ffmpeg (conda-forge, required by torchcodec) ────────────────────────
micromamba install -n residual -c conda-forge "ffmpeg>=6,<8" -y
