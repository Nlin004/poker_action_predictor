#!/bin/bash
#SBATCH --job-name=poker_train
#SBATCH --partition=ice-gpu
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --mem=16G
#SBATCH --output=logs/train_outputs/train_%j.out


source ../.venv/bin/activate

python scripts/train.py \
  --parquet_path data/processed/FULL_PHH_IMPROVED2.parquet \
  --out_dir models/transformerSanityAgain \
  --model transformer \
  --epochs 25 \
  --batch_size 512 \
  --lr 5e-4 \
  # --limit_files 5000 \
