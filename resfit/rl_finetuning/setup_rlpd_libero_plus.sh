#!/usr/bin/env bash
set -euo pipefail

./resfit/lerobot/setup_lerobot.sh
./resfit/libero/setup_libero_plus.sh

pip install wandb einops psutil
