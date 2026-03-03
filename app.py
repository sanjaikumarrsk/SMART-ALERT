from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import io
import torch
from PIL import Image
import numpy as np
from torchvision import transforms
import csv

BASE = Path(__file__).resolve().parent

app = FastAPI(title='Smart Alert Inference')
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def read_label_map_from_eval_csv(model_dir: Path):
    csv_path = model_dir / 'eval_results.csv'
    if not csv_path.exists():
        return None
    label_map = {}
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(row['label'])
            name = row['label_name']
            label_map[name] = idx
    return label_map


def infer_positive_index_from_map(label_map, task_keyword):
    task_keyword = task_keyword.lower()
    for name, idx in label_map.items():
        if task_keyword in name.lower() and 'non' not in name.lower():
            return idx, name
    for name, idx in label_map.items():
        if task_keyword in name.lower():
            return idx, name
    for name, idx in label_map.items():
        if 'non' not in name.lower():
            return idx, name
    return list(label_map.values())[0], list(label_map.keys())[0]


def build_transform(image_size=224):
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


MODEL_CONFIG = {
    'accident': {
        'ts_path': BASE / 'models' / 'accident' / 'model_scripted.pt',
        'threshold': 0.01,
    },
    'fire': {
        'ts_path': BASE / 'models' / 'fire' / 'model_scripted.pt',
        'threshold': 0.01,
    },
    'robbery': {
        'ts_path': BASE / 'models' / 'robbery' / 'model_scripted.pt',
        'threshold': 0.03,
    }
}

DEVICE = torch.device('cpu')
TRANSFORM = build_transform()

# load models
MODELS = {}
POS_INFO = {}
for task, cfg in MODEL_CONFIG.items():
    p = cfg['ts_path']
    if p.exists():
        try:
            m = torch.jit.load(str(p), map_location=DEVICE)
            m.eval()
            MODELS[task] = m
            # read eval CSV if present
            label_map = read_label_map_from_eval_csv(p.parent)
            if label_map:
                idx, name = infer_positive_index_from_map(label_map, task)
                POS_INFO[task] = (idx, name)
            else:
                POS_INFO[task] = (None, None)
        except Exception as e:
            print('Failed to load', p, e)
    else:
        print('Model file not found:', p)


@app.get('/health')
async def health():
    return {'status': 'ok', 'models_loaded': list(MODELS.keys())}


def read_imagefile(file) -> Image.Image:
    image = Image.open(io.BytesIO(file)).convert('RGB')
    return image


@app.post('/predict')
async def predict(file: UploadFile = File(...), task: str = 'all'):
    if task != 'all' and task not in MODELS:
        raise HTTPException(status_code=400, detail='Unknown task')

    contents = await file.read()
    img = read_imagefile(contents)
    inp = TRANSFORM(img).unsqueeze(0).to(DEVICE)

    results = {}
    for tname, model in MODELS.items():
        if task != 'all' and tname != task:
            continue
        with torch.no_grad():
            out = model(inp)
            probs = torch.nn.functional.softmax(out, dim=1).cpu().numpy()[0]

        pos_idx, pos_name = POS_INFO.get(tname, (None, None))
        if pos_idx is None:
            pred_idx = int(np.argmax(probs))
            conf = float(probs[pred_idx])
            label = f'class_{pred_idx}'
        else:
            conf = float(probs[pos_idx])
            label = pos_name

        thresh = MODEL_CONFIG[tname]['threshold']
        alert = conf >= thresh

        results[tname] = {'label': label, 'confidence': conf, 'alert': alert, 'threshold': thresh}

    return JSONResponse(results)


if __name__ == '__main__':
    import uvicorn
    uvicorn.run('app:app', host='0.0.0.0', port=8000, log_level='info')
