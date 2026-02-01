#!/usr/bin/env bash

# --data_dir: Path to the dataset. (Place either TSB-AD-U or TSB-AD-M folder here)
# --output_dir: If you want to see the point-wise scores, specify the directory path. If None, results are not saved.
# --see_loss: if you add, training loss is printed during training.

python main.py \
  --data_dir "data/TSB-AD-M" \
  --patch_size 96 \
  --num_iters 200 \
  --batch_size 512 \
  --lr 1e-4 \
  --seed 2027 \
  --use_revin \
  
