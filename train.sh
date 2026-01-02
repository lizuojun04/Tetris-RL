#!/bin/zsh
CONDA_PATH=$(conda info --base)
source "$CONDA_PATH/etc/profile.d/conda.sh"
conda activate tetris
python -m train.train > ./logs/2026-01-01_15-04/train.log
python -m test.test --test 500 > ./logs/2026-01-01_15-04/test.log
