#!/usr/bin/env bash
set -euo pipefail

./resfit/lerobot/setup_lerobot.sh
./resfit/libero/setup_libero.sh

pip install wandb einops psutil
