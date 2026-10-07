#!/usr/bin/env bash
# Run a python script inside the contactsim env with the runtime fixes this machine needs.
#   bash scripts/run.sh scripts/visualize_env.py --headless --enable_cameras
set -euo pipefail
ENV_PREFIX=${ENV_PREFIX:-$HOME/miniconda3/envs/contactsim}
# Ubuntu 20.04 has no libGLU; without it the RTX MDL backend (libneuray) fails and Kit segfaults.
# Only this one conda-forge lib is exposed so Kit keeps using its own bundled libraries.
export LD_LIBRARY_PATH=$ENV_PREFIX/lib/isaac_extra${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONUNBUFFERED=1
# Driver 535.104 is below Isaac Sim 4.5's RTX minimum (535.129); skip the check (rendering works).
exec "$ENV_PREFIX/bin/python" "$@" --kit_args="--/rtx/verifyDriverVersion/enabled=false"
