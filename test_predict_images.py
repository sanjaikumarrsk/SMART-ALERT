import os
from pathlib import Path
import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image

BASE = Path('e:/smart alert')
MODEL_PATHS = {
    'accident': BASE / 'models' / 'accident' / 'model_scripted.pt',
    'fire': BASE / 'models' / 'fire' / 'model_scripted.pt',
    'robbery': BASE / 'models' / 'robbery' / 'model_scripted.pt',
}

DATASETS = {
    'accident': BASE / 'datasets' / 'accident_nonaccident' / 'val',
    'fire': BASE / 'datasets' / 'fire_nonfire' / 'val',
    'robbery': BASE / 'datasets' / 'robbery_nonrobbery' / 'val',
}

TRANSFORM = transforms.Compose([
    transforms.Resize(int(224 * 1.14)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def load_model(path):
    if not path.exists():
        print('Model missing:', path)
        return None
    m = torch.jit.load(str(path), map_location='cpu')
    m.eval()
    return m

def find_sample_image(val_dir):
    # find first image in any subfolder
    for root, dirs, files in os.walk(val_dir):
        for f in files:
            if f.lower().endswith(('.jpg','.jpeg','.png')):
                return Path(root) / f
    return None

def predict_image(model, img_path):
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img)
    inp = TRANSFORM(pil).unsqueeze(0)
    with torch.no_grad():
        out = model(inp)
        probs = torch.nn.functional.softmax(out, dim=1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))
        conf = float(probs[pred_idx])
    return pred_idx, conf, probs

def main():
    models = {}
    for name, path in MODEL_PATHS.items():
        m = load_model(path)
        models[name] = m

    for task, val_dir in DATASETS.items():
        sample = find_sample_image(val_dir)
        if sample is None:
            print(f'No sample found for {task} in {val_dir}')
            continue
        print(f'[{task}] sample: {sample}')
        m = models.get(task)
        if m is None:
            print('  model not loaded')
            continue
        pred = predict_image(m, sample)
        if pred is None:
            print('  failed to read image')
            continue
        pred_idx, conf, probs = pred
        print(f'  pred_idx={pred_idx} conf={conf:.4f} probs={probs.tolist()}')

if __name__ == "__main__":
    main()
