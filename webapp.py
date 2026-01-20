from flask import Flask, render_template, request, jsonify, send_from_directory, send_file, redirect
from pathlib import Path
import threading
import requests
import io
import time
import cv2
import numpy as np
import csv
import os
try:
    import torch
    from torchvision import transforms
    TORCH_AVAILABLE = True
except:
    TORCH_AVAILABLE = False
    transforms = None
from PIL import Image
from PIL import ImageDraw, ImageFont
import textwrap
from alert_service import send_alert, send_bulk_alert
import json
from datetime import datetime
import base64
import re
try:
    import mcqueen_service
    HAVE_MCQUEEN_LOCAL = True
except Exception:
    mcqueen_service = None
    HAVE_MCQUEEN_LOCAL = False


###############################################
#           FLASK BACKEND CODE               #
###############################################

BASE = Path(__file__).resolve().parent
MODELS_DIR = BASE / 'models'

app = Flask(__name__, static_folder='static', template_folder='templates')

# Users storage
USERS_FILE = BASE / 'users.json'

def load_users():
    try:
        if USERS_FILE.exists():
            with open(USERS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception:
        pass
    return []

def save_users(users):
    try:
        with open(USERS_FILE, 'w', encoding='utf-8') as f:
            json.dump(users, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print('[WEBAPP] Failed to save users:', e)
        return False

def find_user_by_email(email):
    users = load_users()
    for u in users:
        if u.get('email') == email:
            return u
    return None

def read_label_map_from_eval_csv(model_dir: Path):
    csv_path = model_dir / 'eval_results.csv'
    if not csv_path.exists():
        return None
    label_map = {}
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(row['label'])
            label_map[row['label_name']] = idx
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
    names = list(label_map.keys())
    idxs = list(label_map.values())
    return idxs[0], names[0]


def build_transform(image_size=224):
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])


def load_models():
    configs = {}
    for task in ['accident', 'fire', 'robbery']:
        model_dir = MODELS_DIR / task
        ts = model_dir / 'model_scripted.pt'
        best = model_dir / 'best_model.pth'
        configs[task] = {
            'ts': ts if ts.exists() else None,
            'best': best if best.exists() else None,
            'dir': model_dir
        }
    return configs


MODEL_CONFIG = load_models()
MODELS = {}
POS_INFO = {}
TRANSFORM = build_transform() if TORCH_AVAILABLE else None
DEVICE = torch.device('cpu') if TORCH_AVAILABLE else None
LAST_EMERGENCY = {'status': None, 'time': 0.0}
# Require top-confidence >= 0.80 to promote to an emergency and send alerts
EMERGENCY_CONFIDENCE = float(os.environ.get('EMERGENCY_CONFIDENCE', '0.80'))
# Cooldown (seconds) to avoid repeated emergency SMS for the same condition
EMERGENCY_COOLDOWN = float(os.environ.get('EMERGENCY_COOLDOWN', '60'))
# Per-task overrides (robbery stricter)
EMERGENCY_CONFIDENCE_OVERRIDES = {
    'robbery': float(os.environ.get('EMERGENCY_CONFIDENCE_ROBBERY', '0.95'))
}
# Avoid saving many duplicate detection records for the same task in short time
DETECTION_SAVE_WINDOW = int(os.environ.get('DETECTION_SAVE_WINDOW', '60'))
LAST_DETECTION_SAVED = {}

# Load models
if TORCH_AVAILABLE:
    for task, cfg in MODEL_CONFIG.items():
        ts = cfg['ts']
        if ts:
            try:
                m = torch.jit.load(str(ts), map_location=DEVICE)
                m.eval()
                MODELS[task] = m
                label_map = read_label_map_from_eval_csv(cfg['dir'])

                if label_map:
                    POS_INFO[task] = infer_positive_index_from_map(label_map, task)
                else:
                    POS_INFO[task] = (None, None)

                print(f"[MODEL] Loaded {task}")

            except Exception as e:
                print("Model load failed:", task, e)


        # History storage for emergency incidents
        ALERTS_DIR = BASE / 'static' / 'alerts'
        HISTORY_FILE = BASE / 'history.json'
        try:
            ALERTS_DIR.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        # Ensure avatars dir exists
        try:
            (BASE / 'static' / 'avatars').mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        def append_history_record(record: dict):
            try:
                if HISTORY_FILE.exists():
                    with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                        arr = json.load(f)
                else:
                    arr = []
            except Exception:
                arr = []

            # Deduplicate: avoid appending if the most recent record matches task+level within save window
            try:
                if len(arr) > 0:
                    last = arr[0]
                    try:
                        last_time = datetime.fromisoformat(last.get('time').replace('Z', ''))
                    except Exception:
                        last_time = None
                    now = datetime.utcnow()
                    same_task = last.get('task') == record.get('task')
                    same_level = last.get('level') == record.get('level')
                    if same_task and same_level and last_time is not None:
                        delta = (now - last_time).total_seconds()
                        if delta < DETECTION_SAVE_WINDOW:
                            # skip duplicate
                            return
            except Exception:
                pass

            # prepend newest first
            arr.insert(0, record)

            try:
                with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
                    json.dump(arr, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print('[WEBAPP] Failed to write history:', e)


@app.route('/')
def index():
    # Serve the marketing main page first
    return render_template('main.html')


@app.route('/main.html')
def main_html():
    # Friendly route so links to /main.html work
    return render_template('main.html')


@app.route('/index.html')
def index_html():
    return render_template('index.html')

# Serve other pages by friendly routes
@app.route('/how_it_works')
@app.route('/how_it_works.html')
def how_it_works():
    return render_template('how_it_works.html')


@app.route('/guidelines')
@app.route('/guidelines.html')
def guidelines_page():
    return render_template('guidelines.html')

@app.route('/login')
@app.route('/login.html')
def login_page():
    return render_template('login.html')

@app.route('/signup')
@app.route('/signup.html')
def signup_page():
    return render_template('signup.html')


@app.route('/signup', methods=['POST'])
def signup_post():
    data = request.form
    email = data.get('email')
    password = data.get('password')
    confirm = data.get('confirm_password')
    if not email or not password or password != confirm:
        return "Invalid signup data", 400

    if find_user_by_email(email):
        return "User already exists", 400

    user = {k: v for k, v in data.items()}
    users = load_users()
    users.append(user)
    save_users(users)
    return redirect('/login')


@app.route('/login', methods=['POST'])
def login_post():
    data = request.form
    email = data.get('email')
    password = data.get('password')
    user = find_user_by_email(email)
    if user and user.get('password') == password:
        return redirect('/index.html')
    else:
        return redirect('/login')


@app.route('/history', methods=['GET'])
def history():
    try:
        if HISTORY_FILE.exists():
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                arr = json.load(f)
        else:
            arr = []
    except Exception as e:
        print('[WEBAPP] Failed to read history:', e)
        arr = []

    return jsonify(arr)


@app.route('/history_page', methods=['GET'])
def history_page_view():
    return render_template('history_page.html')


@app.route('/profile', methods=['GET'])
def profile_page():
    return render_template('profile.html')


@app.route('/mcqueen')
@app.route('/mcqueen.html')
def mcqueen_page():
    return render_template('mcqueen.html')


@app.route('/api/mcqueen', methods=['POST'])
def api_mcqueen_proxy():
    """Proxy endpoint used by the chat UI. Prefer the local in-process handler
    (mcqueen_service.handle_message) if available; otherwise forward to the
    standalone mcqueen service HTTP API at port 8600."""
    try:
        data = request.get_json(force=True)
    except Exception:
        data = {}
    
    message = data.get('message', '').strip() if isinstance(data, dict) else ''
    
    if not message:
        return jsonify({'reply': 'Please describe what is happening.'}), 200

    # If local handler is available, call it directly
    if HAVE_MCQUEEN_LOCAL:
        try:
            res = mcqueen_service.handle_message(message)
            # ensure JSON serializable dict with a 'reply' key
            if isinstance(res, dict):
                return jsonify(res)
            else:
                return jsonify({'reply': str(res)})
        except Exception as e:
            print(f'[MCQUEEN] Local handler error: {e}')
            return jsonify({'reply': 'Safety assistant unavailable. Call emergency services if needed.'}), 200

    # Otherwise try calling the standalone service over HTTP
    try:
        r = requests.post('http://127.0.0.1:8600/api/mcqueen', json={'message': message}, timeout=4)
        try:
            return jsonify(r.json()), r.status_code
        except Exception:
            return jsonify({'reply': 'Service error. Call emergency services if needed.'}), 200
    except Exception as e:
        print(f'[MCQUEEN] HTTP fallback error: {e}')
        return jsonify({'reply': 'Service unavailable. Call emergency services if needed.'}), 200


@app.route('/api/me', methods=['GET'])
def api_me():
    # Return the first user if no email provided, or the user matching the email
    try:
        users = load_users()
    except Exception:
        users = []
    if not users:
        return jsonify({'error': 'no users'}), 404

    email = request.args.get('email')
    def attach_avatar(u):
        out = dict(u)
        try:
            key = re.sub(r'[^a-z0-9]', '_', (u.get('email') or '').lower())
            p = BASE / 'static' / 'avatars' / f"{key}.png"
            if p.exists():
                out['avatar'] = f"/static/avatars/{key}.png"
        except Exception:
            pass
        return out

    if email:
        u = find_user_by_email(email)
        if not u:
            return jsonify({'error': 'user not found'}), 404
        return jsonify(attach_avatar(u))
    # default: return first user
    return jsonify(attach_avatar(users[0]))


@app.route('/api/update_profile', methods=['POST'])
def api_update_profile():
    # Accepts JSON body with updated fields. Matches by email and updates user record.
    data = None
    try:
        data = request.get_json()
    except Exception:
        data = None
    if not data:
        return jsonify({'error': 'missing json body'}), 400

    email = data.get('email')
    if not email:
        return jsonify({'error': 'missing email'}), 400

    users = load_users()
    updated = False
    for i, u in enumerate(users):
        if u.get('email') == email:
            users[i].update(data)
            updated = True
            break

    if not updated:
        # append new user (if email not found)
        users.append(data)

    ok = save_users(users)
    if ok:
        return jsonify({'ok': True})
    else:
        return jsonify({'error': 'save failed'}), 500


@app.route('/logout')
def logout():
    # Simple logout redirect (no session implemented)
    return redirect('/login')


@app.route('/api/upload_avatar', methods=['POST'])
def api_upload_avatar():
    try:
        if 'avatar' not in request.files:
            return jsonify({'error': 'missing file'}), 400
        f = request.files['avatar']
        email = request.form.get('email') or request.args.get('email')
        if not email:
            return jsonify({'error': 'missing email'}), 400

        # sanitize email to safe filename
        safe = re.sub(r'[^a-z0-9]', '_', email.lower())
        avatars_dir = BASE / 'static' / 'avatars'
        try:
            avatars_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        save_path = avatars_dir / f"{safe}.png"
        # Save uploaded file (Pillow can be used to normalize, but simple save is fine)
        f.save(str(save_path))
        return jsonify({'ok': True, 'url': f'/static/avatars/{safe}.png'})
    except Exception as e:
        print('[WEBAPP] avatar upload failed:', e)
        return jsonify({'error': 'upload failed'}), 500


@app.route('/history/download', methods=['GET'])
def history_download():
    # Generate a PDF containing each history record with image and details
    try:
        if HISTORY_FILE.exists():
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                arr = json.load(f)
        else:
            arr = []
    except Exception:
        arr = []

    # Use Pillow to compose pages and save as PDF in-memory
    pages = []
    PAGE_W, PAGE_H = 1240, 1754
    margin = 40
    # Try to load a truetype font for nicer rendering, fallback to default
    try:
        header_font = ImageFont.truetype("arial.ttf", 36)
        title_font = ImageFont.truetype("arial.ttf", 28)
        text_font = ImageFont.truetype("arial.ttf", 20)
    except Exception:
        header_font = ImageFont.load_default()
        title_font = ImageFont.load_default()
        text_font = ImageFont.load_default()

    for rec in arr:
        page = Image.new('RGB', (PAGE_W, PAGE_H), color=(255, 255, 255))
        draw = ImageDraw.Draw(page)

        # header
        draw.text((margin, margin // 2), "SMART ALERT - History Export", font=header_font, fill=(10, 10, 10))
        draw.line((margin, margin + 28, PAGE_W - margin, margin + 28), fill=(200, 200, 200), width=2)

        # left area: image (fixed box)
        img_box_w = 520
        img_box_h = 520

        img = None
        # Try to load image from disk
        name = rec.get('name')
        if name:
            img_path = ALERTS_DIR / name
            if img_path.exists():
                try:
                    img = Image.open(str(img_path)).convert('RGB')
                except Exception:
                    img = None

        # Fallback to base64 image in JSON
        if img is None and rec.get('image_b64'):
            try:
                b = base64.b64decode(rec.get('image_b64'))
                img = Image.open(io.BytesIO(b)).convert('RGB')
            except Exception:
                img = None

        if img is not None:
            # fit into box and center inside box
            img.thumbnail((img_box_w, img_box_h), Image.Resampling.LANCZOS)
            ix = margin
            iy = margin + 40 + (img_box_h - img.height) // 2
            # draw image background box
            draw.rectangle([ix - 4, iy - 4, ix + img_box_w + 4, iy + img_box_h + 4], outline=(220, 220, 220), width=1)
            page.paste(img, (ix + (img_box_w - img.width) // 2, iy + (img_box_h - img.height) // 2))
        else:
            # placeholder rectangle
            ix = margin
            iy = margin + 40
            draw.rectangle([ix, iy, ix + img_box_w, iy + img_box_h], outline=(200, 200, 200), width=2)
            draw.text((ix + 12, iy + 12), 'No image', fill=(100, 100, 100), font=text_font)

        # right area: details
        tx = margin + img_box_w + 40
        ty = margin + 40
        max_w = PAGE_W - tx - margin
        line_h = 28

        # Compose detail lines with wrapping
        details = [
            ("Task", rec.get('task', '')),
            ("Level", rec.get('level', '')),
            ("Confidence", f"{int(float(rec.get('confidence', 0))*100)}%"),
            ("Location", rec.get('location', '')),
            ("Time", rec.get('time', '')),
            ("Filename", rec.get('name', ''))
        ]

        cur_y = ty
        for label_text, value in details:
            # draw label
            draw.text((tx, cur_y), f"{label_text}:", font=title_font, fill=(30, 30, 30))
            # wrap value
            wrapped = textwrap.wrap(str(value), width=40)
            # draw first line next to label
            if len(wrapped) > 0:
                draw.text((tx + 140, cur_y), wrapped[0], font=text_font, fill=(40, 40, 40))
            cur_y += line_h
            # draw remaining wrapped lines
            for wln in wrapped[1:]:
                draw.text((tx + 140, cur_y), wln, font=text_font, fill=(40, 40, 40))
                cur_y += line_h
            cur_y += 6

        pages.append(page)

    if len(pages) == 0:
        # return empty PDF
        empty = Image.new('RGB', (PAGE_W, PAGE_H), color=(255, 255, 255))
        pages = [empty]

    bio = io.BytesIO()
    try:
        pages[0].save(bio, format='PDF', save_all=True, append_images=pages[1:])
        bio.seek(0)
        return send_file(bio, mimetype='application/pdf', as_attachment=True, download_name='history_export.pdf')
    except Exception as e:
        print('[WEBAPP] Failed to generate PDF:', e)
        return jsonify({'error': 'Failed to generate PDF'}), 500


@app.route('/history/download_record', methods=['GET'])
def history_download_record():
    name = request.args.get('name')
    if not name:
        return jsonify({'error': 'Missing name parameter'}), 400

    try:
        if HISTORY_FILE.exists():
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                arr = json.load(f)
        else:
            arr = []
    except Exception:
        arr = []

    rec = None
    for r in arr:
        if r.get('name') == name:
            rec = r
            break

    if rec is None:
        return jsonify({'error': 'Record not found'}), 404

    # Generate a single-page PDF for this record (reuse layout from history_download)
    PAGE_W, PAGE_H = 1240, 1754
    margin = 40
    try:
        header_font = ImageFont.truetype("arial.ttf", 36)
        title_font = ImageFont.truetype("arial.ttf", 28)
        text_font = ImageFont.truetype("arial.ttf", 20)
    except Exception:
        header_font = ImageFont.load_default()
        title_font = ImageFont.load_default()
        text_font = ImageFont.load_default()

    page = Image.new('RGB', (PAGE_W, PAGE_H), color=(255, 255, 255))
    draw = ImageDraw.Draw(page)
    draw.text((margin, margin // 2), "SMART ALERT - Record Export", font=header_font, fill=(10, 10, 10))
    draw.line((margin, margin + 28, PAGE_W - margin, margin + 28), fill=(200, 200, 200), width=2)

    img_box_w = 520
    img_box_h = 520

    img = None
    if rec.get('name'):
        img_path = ALERTS_DIR / rec.get('name')
        if img_path.exists():
            try:
                img = Image.open(str(img_path)).convert('RGB')
            except Exception:
                img = None

    if img is None and rec.get('image_b64'):
        try:
            b = base64.b64decode(rec.get('image_b64'))
            img = Image.open(io.BytesIO(b)).convert('RGB')
        except Exception:
            img = None

    if img is not None:
        img.thumbnail((img_box_w, img_box_h), Image.Resampling.LANCZOS)
        ix = margin
        iy = margin + 40 + (img_box_h - img.height) // 2
        draw.rectangle([ix - 4, iy - 4, ix + img_box_w + 4, iy + img_box_h + 4], outline=(220, 220, 220), width=1)
        page.paste(img, (ix + (img_box_w - img.width) // 2, iy + (img_box_h - img.height) // 2))
    else:
        ix = margin
        iy = margin + 40
        draw.rectangle([ix, iy, ix + img_box_w, iy + img_box_h], outline=(200, 200, 200), width=2)
        draw.text((ix + 12, iy + 12), 'No image', fill=(100, 100, 100), font=text_font)

    tx = margin + img_box_w + 40
    ty = margin + 40
    line_h = 28
    details = [
        ("Task", rec.get('task', '')),
        ("Level", rec.get('level', '')),
        ("Confidence", f"{int(float(rec.get('confidence', 0))*100)}%"),
        ("Location", rec.get('location', '')),
        ("Time", rec.get('time', '')),
        ("Filename", rec.get('name', ''))
    ]
    cur_y = ty
    for label_text, value in details:
        draw.text((tx, cur_y), f"{label_text}:", font=title_font, fill=(30, 30, 30))
        wrapped = textwrap.wrap(str(value), width=40)
        if len(wrapped) > 0:
            draw.text((tx + 140, cur_y), wrapped[0], font=text_font, fill=(40, 40, 40))
        cur_y += line_h
        for wln in wrapped[1:]:
            draw.text((tx + 140, cur_y), wln, font=text_font, fill=(40, 40, 40))
            cur_y += line_h
        cur_y += 6

    bio = io.BytesIO()
    try:
        page.save(bio, format='PDF')
        bio.seek(0)
        return send_file(bio, mimetype='application/pdf', as_attachment=True, download_name=f'{name}.pdf')
    except Exception as e:
        print('[WEBAPP] Failed to generate single-record PDF:', e)
        return jsonify({'error': 'Failed to generate PDF'}), 500


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            data = request.get_data()
            img = Image.open(io.BytesIO(data)).convert('RGB')
        else:
            f = request.files['image']
            data = f.read()
            img = Image.open(io.BytesIO(data)).convert('RGB')

        inp = TRANSFORM(img).unsqueeze(0).to(DEVICE).float()

        results = {}
        overall_alerts = []

        for task, model in MODELS.items():
            with torch.no_grad():
                out = model(inp)

            probs = torch.nn.functional.softmax(out, dim=1).cpu().numpy()[0]
            pos_idx, pos_name = POS_INFO[task]

            conf = float(probs[pos_idx])
            label = pos_name
            # Determine per-task required confidence for detection/promotion
            required = EMERGENCY_CONFIDENCE_OVERRIDES.get(task, EMERGENCY_CONFIDENCE)
            alert = conf >= required

            if alert:
                overall_alerts.append({"task": task, "confidence": conf})

            results[task] = {"label": label, "confidence": conf, "alert": alert}

        overall = {"status": "SAFE", "alerts": []}

        # Persist per-model detection records, but avoid duplicates within DETECTION_SAVE_WINDOW
        if overall_alerts:
            for a in overall_alerts:
                try:
                    det_now = int(time.time())
                    last = int(LAST_DETECTION_SAVED.get(a['task'], 0))
                    # Skip saving if we saved recently for this task
                    if det_now - last < DETECTION_SAVE_WINDOW:
                        continue

                    det_fname = f"{a['task']}_{det_now}_det.jpg"
                    det_path = ALERTS_DIR / det_fname
                    with open(det_path, 'wb') as df:
                        df.write(data)
                    public_path = f"/static/alerts/{det_fname}"

                    # record that we saved recently for this task
                    LAST_DETECTION_SAVED[a['task']] = det_now
                except Exception as e:
                    print('[WEBAPP] Failed to save detection image:', e)
                    public_path = None

                rec = {
                    'task': a['task'],
                    'time': datetime.utcnow().isoformat() + 'Z',
                    'location': 'THALAVAPALAYAM',
                    'confidence': float(a.get('confidence', 0.0)),
                    'path': public_path,
                    'name': det_fname if public_path else None,
                    'image_b64': base64.b64encode(data).decode('ascii') if data is not None else None,
                    'level': 'detection'
                }
                append_history_record(rec)

        if overall_alerts:
            top = max(overall_alerts, key=lambda x: x["confidence"])
            max_conf = float(top.get("confidence", 0.0))
            overall["alerts"] = overall_alerts
            overall["top_confidence"] = max_conf
            # Only set emergency status if top confidence meets the per-task threshold
            required_top = EMERGENCY_CONFIDENCE_OVERRIDES.get(top['task'], EMERGENCY_CONFIDENCE)
            if max_conf >= required_top:
                overall["status"] = top["task"].upper()
                overall["emergency"] = True
            else:
                overall["status"] = "SAFE"
                overall["emergency"] = False

            now = time.time()
            # Send alert only when we have an emergency and status changed or cooldown expired
            if overall.get("emergency"):
                if LAST_EMERGENCY["status"] != overall["status"] or now - LAST_EMERGENCY["time"] > EMERGENCY_COOLDOWN:
                    # persist alert image
                    filename = f"{top['task']}_{int(now)}.jpg"
                    try:
                        img_path = ALERTS_DIR / filename
                        with open(img_path, 'wb') as fimg:
                            fimg.write(data)
                        public_path = f"/static/alerts/{filename}"
                    except Exception as e:
                        print('[WEBAPP] Failed to save alert image:', e)
                        public_path = None

                    # send alert (sms/email) and capture result
                    send_result = None
                    try:
                        send_result = send_alert(
                            incident_type=top["task"],
                            location="THALAVAPALAYAM",
                            confidence=max_conf,
                            image_bytes=data,
                            image_filename=filename,
                            send_email=True
                        )
                        print(f'[WEBAPP] send_alert result: {send_result}')
                    except Exception as e:
                        print('[WEBAPP] send_alert failed:', e)
                        send_result = {'error': str(e)}

                    # append to history (include send_result)
                    rec = {
                        'task': top['task'],
                        'time': datetime.utcnow().isoformat() + 'Z',
                        'location': 'THALAVAPALAYAM',
                        'confidence': float(max_conf),
                        'path': public_path,
                        'name': filename,
                        'image_b64': base64.b64encode(data).decode('ascii') if data is not None else None,
                        'level': 'emergency',
                        'alert_result': send_result,
                        'suppressed': False
                    }
                    append_history_record(rec)

                    LAST_EMERGENCY["status"] = overall["status"]
                    LAST_EMERGENCY["time"] = now
                else:
                    print('[WEBAPP] Emergency detected but suppressed to avoid repeat SMS (cooldown).')
                    # even if suppressed, store a history record indicating suppression
                    try:
                        filename = f"{top['task']}_{int(now)}_supp.jpg"
                        img_path = ALERTS_DIR / filename
                        with open(img_path, 'wb') as fimg:
                            fimg.write(data)
                        public_path = f"/static/alerts/{filename}"
                    except Exception:
                        public_path = None
                    rec = {
                        'task': top['task'],
                        'time': datetime.utcnow().isoformat() + 'Z',
                        'location': 'THALAVAPALAYAM',
                        'confidence': float(max_conf),
                        'path': public_path,
                        'name': filename if public_path else None,
                        'image_b64': base64.b64encode(data).decode('ascii') if data is not None else None,
                        'level': 'emergency',
                        'alert_result': None,
                        'suppressed': True
                    }
                    append_history_record(rec)

        return jsonify({"overall": overall, "per_model": results})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Start mobile camera thread

    # Start Flask server
    app.run(host="0.0.0.0", port=8000,debug=False)
