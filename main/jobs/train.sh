#!/bin/bash
#SBATCH --job-name=poker_train
#SBATCH --partition=ice-gpu
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --mem=16G
#SBATCH --output=logs/train_%j.out


source ../.venv/bin/activate

python scripts/train_next_action.py \
  --mode parquet \
  --parquet_path data/processed/pluribus1.parquet \
  --limit_files 1000 \
  --out_dir models/MLP1 \
  --model lstm \
  --epochs 25 \
  --batch_size 512