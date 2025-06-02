#!/bin/bash

# Define arrays
models=("erpp" "stlstm" "rnn" "gru" "transformer" "decoder")
embeds=("hier" "ctle" "fourier" "teaser")
datasets=("pek" "taxi")

# Device setting
device=0

# Loop over all combinations
for model in "${models[@]}"; do
  for embed in "${embeds[@]}"; do
    for dataset in "${datasets[@]}"; do
      echo "Running: model=$model, embed=$embed, dataset=$dataset"
      python run.py --pre_model_name "$model" --embed_name "$embed" --task_epoch 20 --embed_epoch 10 --device "$device" --dataset "$dataset"
    done
  done
done
