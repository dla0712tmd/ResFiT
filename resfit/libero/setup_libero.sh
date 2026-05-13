# Get the directory of this script and navigate to repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEPS_DIR="$REPO_ROOT/deps"

# Create deps directory if it doesn't exist
mkdir -p "$DEPS_DIR"

# Git clone robosuite into deps directory
git clone https://github.com/ARISE-Initiative/robosuite "$DEPS_DIR/robosuite"
git -C "$DEPS_DIR/robosuite" checkout v1.4.0

# Install robosuite
python -m pip install -e "$DEPS_DIR/robosuite"

# Git clone LIBERO into deps directory
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git "$DEPS_DIR/libero"
git -C "$DEPS_DIR/libero" checkout 8f1084e3132a39270c3a13ebe37270a43ece2a01

# Install LIBERO
python -m pip install -e "$DEPS_DIR/libero"
python -m pip install bddl==1.0.1 easydict==1.9 future==0.18.2 cloudpickle==2.1.0 "gym==0.25.2"

# Workaround for editable install MAPPING bug: add path directly via .pth file
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
echo "$DEPS_DIR/libero" > "$SITE_PACKAGES/libero-path.pth"

# Write the LIBERO config directly into a package-specific directory so it does
# not conflict with the LIBERO-plus config (both default to ~/.libero otherwise).
python - "$DEPS_DIR/libero" <<'PYEOF'
import os, sys, yaml
libero_root = os.path.join(sys.argv[1], "libero", "libero")
config_dir  = os.path.join(sys.argv[1], ".libero_config")
os.makedirs(config_dir, exist_ok=True)
config = {
    "benchmark_root": libero_root,
    "bddl_files":     os.path.join(libero_root, "bddl_files"),
    "init_states":    os.path.join(libero_root, "init_files"),
    "datasets":       os.path.join(sys.argv[1], "libero", "datasets"),
    "assets":         os.path.join(libero_root, "assets"),
}
with open(os.path.join(config_dir, "config.yaml"), "w") as f:
    yaml.dump(config, f)
print(f"[setup_libero] config written → {config_dir}/config.yaml")
PYEOF

# Install a couple of dependencies
python -m pip install gymnasium==1.1.1

# Install PyOpenGL-accelerate
python -m pip install PyOpenGL-accelerate

# Original additional installs
python -m pip install ipdb pyserial deepdiff matplotlib

python -m pip install -U "numba>=0.59" "llvmlite>=0.42"

python -m pip install tabulate

python -m pip install torchrl==0.8.0 tensordict==0.8.2 torchcodec==0.4.0

python -m pip install mujoco==3.3.2 "protobuf>4.21.0,<5" diffusers==0.33.1 llvmlite==0.42.0 multidict==6.0.5 numba==0.59.1

conda install -n residual -c conda-forge "ffmpeg>=6,<8" -y

# Upgrade to a Numba that supports NumPy 2.x (and its llvmlite)
pip install --upgrade --no-cache-dir "numba>=0.60" "llvmlite>=0.44"

pip install draccus==0.10.0
