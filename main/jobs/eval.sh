source ../.venv/bin/activate

python scripts/infer_next_action.py \
  --model_path models/test1/best_model.pt \
  --phh_file data/raw/phh-dataset/data/pluribus/30/0.phh \
  --model_type lstm