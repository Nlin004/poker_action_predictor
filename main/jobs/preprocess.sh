# run this from main!
 
source ../.venv/bin/activate


python scripts/preprocess_phh_dataset.py \
  --input_dir data/raw/phh-dataset/data/pluribus \
  --output data/processed/pluribus1.parquet \
  --limit 10000