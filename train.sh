#!/bin/zsh
CONDA_PATH=$(conda info --base)
source "$CONDA_PATH/etc/profile.d/conda.sh"
conda activate tetris
python -m train.train > ./logs/2025-12-31_19-54/train.log
python -m test.test --test 500 > ./logs/2025-12-31_19-54/test.log
