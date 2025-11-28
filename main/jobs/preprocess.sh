# run this from main!
 
source ../.venv/bin/activate


# python scripts/preprocess_phh_dataset.py \
#   --input_dir data/raw/phh-dataset/data/handhq/ABS-2009-07-01_2009-07-23_50NLH_OBFU/0.5 \
#   --output data/processed/handhq.parquet \
#   --limit 10000


python scripts/preprocess_new.py \
  --input_dir data/raw/phh-dataset/data/handhq/ABS-2009-07-01_2009-07-23_50NLH_OBFU/0.5 \
  --output data/processed/handhq.parquet \
  --limit 10000