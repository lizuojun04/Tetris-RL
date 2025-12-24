#!/bin/zsh
CONDA_PATH=$(conda info --base)
source "$CONDA_PATH/etc/profile.d/conda.sh"
conda activate tetris
python -m train.train > train.log
python -m test.test --test 500 > test.log
