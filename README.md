# Emergency Detectors - Training

This project trains binary image detectors for three emergency types: **fire**, **accident**, and **robbery**. The dataset layout (expected) is:

```
datasets/
  fire_nonfire/
    train/
      fire/
      non_fire/
    val/
      fire/
      non_fire/
  accident_nonaccident/
    train/
      accident/
      non_accident/
    val/
      accident/
      non_accident/
  robbery_nonrobbery/
    train/
      robbery/
      non_robbery/
    val/
      robbery/
      non_robbery/
```

Prerequisites
- Create a Python virtualenv and install requirements:

PowerShell:
```
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r "e:\smart alert\requirements.txt"
```

Training
- To train a binary detector for one task (e.g., `fire`):

PowerShell example:
```
python "e:\smart alert\train.py" --dataset-dir "e:\smart alert\datasets\fire_nonfire" --epochs 8 --batch-size 32 --output-dir "e:\smart alert\models\fire"
```

- Train `accident` detector:
```
python "e:\smart alert\train.py" --dataset-dir "e:\smart alert\datasets\accident_nonaccident" --epochs 8 --batch-size 32 --output-dir "e:\smart alert\models\accident"
```

- Train `robbery` detector:
```
python "e:\smart alert\train.py" --dataset-dir "e:\smart alert\datasets\robbery_nonrobbery" --epochs 8 --batch-size 32 --output-dir "e:\smart alert\models\robbery"
```

What you get
- A saved model checkpoint `best_model.pth` in the `--output-dir`.
- Training/validation accuracy printed each epoch.

Next steps
- Evaluate with a hold-out test set and compute precision/recall.
- Convert model to an inference service and add a confidence threshold to trigger alerts to authorities.
