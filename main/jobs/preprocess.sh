#!/bin/bash
#SBATCH -J preprocess_phh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=16GB
#SBATCH -t 12:00:00
#SBATCH -o logs/preprocess_outputs/preprocess_%j.out
 
source ../.venv/bin/activate

python scripts/preprocess_phh_dataset.py \
  --input_dir data/raw/phh-dataset/data \
  --output data/processed/FULL_PHH_IMPROVED2.parquet