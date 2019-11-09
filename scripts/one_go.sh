#!/bin/bash -i

export TF_CPP_MIN_LOG_LEVEL=3

set +x
conda deactivate
conda activate tensorflow
set -x

python split_dataset.py
python initial_training.py
python train_with_pruning.py

set +x
conda deactivate
conda activate tfkerassurgeon
set -x

python remove_filters.py

set +x
conda deactivate
conda activate tensorflow
set -x

python measure.py

set +x