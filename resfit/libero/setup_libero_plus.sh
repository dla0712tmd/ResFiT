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
    libexpat1 libfontconfig1-dev libpython3-stdlib libmagickwand-dev

# ── 2. LIBERO-plus (replaces standard LIBERO) ──────────────────────────────
# Git clone robosuite into deps directory
git clone https://github.com/ARISE-Initiative/robosuite "$DEPS_DIR/robosuite"
git -C "$DEPS_DIR/robosuite" checkout v1.4.0

# Install robosuite
python -m pip install -e "$DEPS_DIR/robosuite"

# Git clone LIBERO-plus into deps directory
[ -d "$DEPS_DIR/libero" ] || \
    git clone https://github.com/sylvestf/LIBERO-plus.git "$DEPS_DIR/libero"
python -m pip install -e "$DEPS_DIR/libero"
python -m pip install -r "$DEPS_DIR/libero/extra_requirements.txt"
python -m pip install \
    bddl==1.0.1 easydict==1.9 future==0.18.2 cloudpickle==2.1.0 "gym==0.25.2"

# Workaround for editable-install path-mapping bug
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
echo "$DEPS_DIR/libero" > "$SITE_PACKAGES/libero-path.pth"

# Pre-generate libero config to avoid interactive prompt on first import
echo "N" | python -c "from libero.libero import benchmark" 2>/dev/null || true

# ── 3. LIBERO-plus extra assets (objects, textures, init-states, BDDL files) ─
python -m pip install huggingface_hub
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
# The zip stores files under a deep internal prefix; strip it on extraction.
with zipfile.ZipFile(zip_path) as z:
    prefix = next(
        n for n in z.namelist()
        if n.endswith("/assets/") and "LIBERO-plus" in n
    )
    for member in z.infolist():
        if not member.filename.startswith(prefix):
            continue
        rel = member.filename[len(prefix):]
        if not rel:
            continue
        dst = assets_dst / rel
        if member.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(z.read(member))
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
    "numba>=0.60" "llvmlite>=0.44" \
    tabulate ipdb pyserial deepdiff matplotlib draccus==0.10.0

# ── 5. torchrl / tensordict (PyPI-only; not on the PyTorch whl server) ──────
# Use --no-deps to prevent pip from pulling torch from PyPI and overwriting
# the cu128 build. Install only the non-torch transitive deps explicitly.
python -m pip install torchrl==0.8.0 tensordict==0.8.2 --no-deps
python -m pip install cloudpickle importlib_metadata orjson packaging

# ── 6. torchcodec cu128 build (must come from PyTorch whl, not PyPI) ────────
python -m pip install torchcodec==0.11.1 --index-url https://download.pytorch.org/whl/cu128
# NPP (NVIDIA Performance Primitives) required by the cu128 torchcodec build.
# torchcodec from the PyTorch WHL server declares no Python deps (unlike PyPI),
# so nvidia-npp-cu12 is not auto-installed as a transitive dep.
# Additionally, PyTorch does NOT preload NPP at startup (torch/__init__.py only
# preloads cublas/cudnn/cuda_nvrtc/cuda_runtime/etc.), so libnppicc.so.12 is
# never added to the dynamic linker cache and ctypes.CDLL cannot find it by name.
# Fix: install the package and symlink the .so into the conda env's lib dir.
python -m pip install nvidia-npp-cu12
# torchcodec (cu128) needs NPP libs at runtime, but PyTorch does not preload them
# (torch.__init__._preload_cuda_deps only covers cublas/cudnn/cuda_nvrtc/etc.).
# Register the path with ldconfig so the dynamic linker can find libnpp*.so.12.
echo "$SITE_PACKAGES/nvidia/npp/lib" > /etc/ld.so.conf.d/nvidia-npp-cu12.conf
ldconfig

# ── 7. ffmpeg (conda-forge, required by torchcodec) ────────────────────────
conda install -n residual -c conda-forge "ffmpeg>=6,<8" -y
