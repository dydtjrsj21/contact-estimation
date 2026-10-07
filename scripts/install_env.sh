#!/usr/bin/env bash
# Reproducible install of Isaac Sim 4.5 + Isaac Lab 2.1.1 into the `contactsim` conda env.
#
# This machine is Ubuntu 20.04 (glibc 2.31), but the Isaac Sim 4.5 pip wheels are tagged
# manylinux_2_34, so pip refuses them. The binaries do run here, so we download the wheels for the
# 2_34 platform, re-tag the *file names* to 2_31 and install from that local folder.
#
# Usage: bash scripts/install_env.sh [wheel_cache_dir]
set -euo pipefail

ENV_NAME=contactsim
CONDA=${CONDA:-$HOME/miniconda3/bin/conda}
PROJECT_ROOT=$(cd "$(dirname "$0")/../.." && pwd)        # .../project
ISAACLAB_DIR=$PROJECT_ROOT/IsaacLab
WHEELS=${1:-$PROJECT_ROOT/.isaacsim_wheels}

if ! "$CONDA" env list | grep -q "^$ENV_NAME "; then
    # conda-forge only: avoids the Anaconda default-channel ToS prompt
    "$CONDA" create -y -n $ENV_NAME --override-channels -c conda-forge python=3.10
fi
PY=$("$CONDA" run -n $ENV_NAME which python)
PIP="$PY -m pip"

$PIP install --upgrade pip
$PIP install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

# --- Isaac Sim 4.5 (glibc workaround) ---
mkdir -p "$WHEELS"
$PIP download "isaacsim[all,extscache]==4.5.0" \
    --extra-index-url https://pypi.nvidia.com \
    --platform manylinux_2_34_x86_64 --python-version 3.10 --only-binary=:all: \
    --no-deps -d "$WHEELS"
# isaacsim[all,extscache] expands to the isaacsim-* sub-packages (+ omniverse-kit, the only other
# 2_34-tagged dependency); fetch them explicitly without deps
$PIP download --no-deps -d "$WHEELS" \
    --extra-index-url https://pypi.nvidia.com \
    --platform manylinux_2_34_x86_64 --python-version 3.10 --only-binary=:all: \
    $(for p in app asset benchmark code-editor core cortex example extscache-kit extscache-kit-sdk \
        extscache-physics gui kernel replicator rl robot robot-motion robot-setup ros1 ros2 sensor \
        storage template test utils; do echo "isaacsim-$p==4.5.0"; done) \
    omniverse-kit==106.5.0.162521
for f in $(ls "$WHEELS"/*manylinux_2_34_x86_64.whl 2>/dev/null); do
    mv "$f" "${f/manylinux_2_34_x86_64/manylinux_2_31_x86_64}"
done
$PIP install "isaacsim[all,extscache]==4.5.0" --find-links "$WHEELS" --extra-index-url https://pypi.nvidia.com

# --- Isaac Lab 2.1.1 (supports Isaac Sim 4.5) ---
if [ ! -d "$ISAACLAB_DIR" ]; then
    git clone --depth 1 --branch v2.1.1 https://github.com/isaac-sim/IsaacLab.git "$ISAACLAB_DIR"
fi
# flatdict (isaaclab dep) needs pkg_resources at build time
$PIP install "setuptools<70" wheel
$PIP install --no-build-isolation flatdict==4.0.1
for pkg in isaaclab isaaclab_assets; do
    $PIP install -e "$ISAACLAB_DIR/source/$pkg"
done

# --- runtime fix: Ubuntu 20.04 has no libGLU (needed by the RTX MDL backend); see scripts/run.sh
"$CONDA" install -y -n $ENV_NAME --override-channels -c conda-forge libglu
ENV_PREFIX=$(dirname "$(dirname "$PY")")
mkdir -p "$ENV_PREFIX/lib/isaac_extra"
ln -sf "$ENV_PREFIX/lib/libGLU.so.1" "$ENV_PREFIX/lib/isaac_extra/libGLU.so.1"

# --- project extras ---
$PIP install matplotlib imageio imageio-ffmpeg h5py
$PIP install -e "$PROJECT_ROOT/contact_estimation"

echo "Done. Wheel cache at $WHEELS (~4 GB) can be deleted."
