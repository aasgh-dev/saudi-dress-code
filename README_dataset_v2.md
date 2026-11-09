# Dataset v2 and training pipeline

Multi-label classifier for `[thobe, ghutra, igal]` on YOLOv8n person crops (EfficientNet-B0).

## Setup (Anaconda Prompt)

```
conda env create -p .\env -f environment.yml
conda activate .\env
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## Try the model

```
python try_model.py --source photo.jpg --weights models\merged_best.pt
python try_model.py --source camera --weights models\merged_best.pt
```

## Retrain

```
python train_merged.py
```

Uses `data/dataset_merged` (1,076 crops: train 758, val 213, test 105). The `test` split has not been evaluated yet.
Results of the last run are in `models/merged_val_results.json`.

## Notes

- `clean_*.py`, `merge_dataset.py` and `collect_*.py` rebuild the dataset from raw folders that are not in this branch.
- `models/trial_best.pt` is the first trial on the original images only.
