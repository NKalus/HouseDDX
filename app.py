# Project layout
# ├── app.py
# ├── requirements.txt
# ├── render.yaml
# ├── Procfile              # (optional if you use render.yaml startCommand)
# ├── templates/
# │   ├── index.html
# │   └── error.html
# └── static/
#     └── style.css

from __future__ import annotations

# --- keep TF tame on tiny dynos BEFORE importing it ---
import os
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import io
import logging
import random
import threading
import traceback
from typing import Optional

from flask import Flask, render_template, request, abort
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np

app = Flask(__name__)

app.config.update(
    SECRET_KEY=os.getenv("SECRET_KEY", "dev"),
    UPLOAD_FOLDER=os.getenv("UPLOAD_FOLDER", "uploads"),
    MAX_CONTENT_LENGTH=10 * 1024 * 1024,  # 10 MB
)
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
log = logging.getLogger("house-ddx")

# ------------------------------
# Long, House-appropriate diagnosis bank
# ------------------------------
diagnosis_bank = {
    "rash": [
        "Hyperreactive dermoepidermal exuberance syndrome (your skin is being an idior)",
        "Vesiculopathic keratinocyte rebellion (your skin cells are evil)",
        "Transient necrotizing flambodermatitis (because normal rashes are too mainstream)",
        "Autoimmune blistering with photophobic angst (your instagram sucks)",
        "Idiopathic itchosaurus complex (you are half-dinosaur)",
        "Contact-induced stupidity pox",
        "Subdermal drama eruption (your skin can't handle your awful personality)",
        "Multifocal epidermal tantrum disorder (too much video game rage)",
        "Heat-seeking dermatitis (take a cold shower and eat a pastrami sandwish and you'll be fine by morning)",
        "Emotionally unstable skin syndrome (your ex poisoned you)",
        "Neurotic scratch cycle disorder (you are part orangutan. You never met your father did you?)",
        "Frantic blister regression disorder (your skin tried to cancel itself)",
    ],
    "swelling": [
        "Interstitial idiopathic puffification disorder (you have fish DNA)",
        "Chronic edematosis of the hypochondrium (your brain is on fire)",
        "Pseudolymphatic egomania (your ego is retaining water)",
        "Ballooning of the moron lobe",
        "Aquapocalyptic infiltration disorder (your body is drowning itself. We need to blow-dry your spinal cord.)",
        "Venolymphatic dairy retention (your veins have milky blood)",
        "Passive-aggressive immuno-accumulation (your immune system only likes you as a friend)",
    ],
    "bruise": [
        "Contusional dumbassity (you don't have cancer you're an idiot with a bruise)",
        "Alcohol-induced capillary vasovagal fragility",
        "Ecchymotic memory recall lesion (Try to back with your ex)",
        "Iatrogenic apathy (I'm playing my gameboy. Go away)",
        "Shame compression under skin (emotional bruising now visible to all)",
        "Denial-induced discolorative patterning",
        "Platelet-induced regret patch",
        "Vascular apology formation (the bruise says what words never will)",
        "Drama imprint syndrome",
        "Purpura of avoidable stupidity",
        "Trauma seepage of the brainless variety (you're a drama queen. And an idiot)",
    ],
    "growth": [
        "Benign attention-seeking pseudotumor (you don't have cancer you moron it's a pimple. Buy some acne cream)",
        "Overcompensation fibroid (you have cancer from too many boner pills)",
        "Delusionoma (idk just go away I need to go bother Cuddy (she took away my cocaine spoon, totally uncalled for))",
        "Hope cyst with a low probability of fulfillment",
        "Cognitive tumor of bad decisions (you have idiot cancer, you need a brain transplant)",
        "Overconfidence blob (resistant to feedback)",
        "Stagnant ambition carcinoma (you have cancer from the resin on your unemployment checks)",
    ],
    "burn": [
        "First-degree idiocy scorch",
        "Consequential thermogenic moron disease",
        "Pyrodramatitis (ignited by emotional combustion)",
        "Open-flame barbeculopathy (your hot dog had granulomatosis)",
        "Moron-induced keratinolysis",
        "Stupid Baby Syndrome (you need to have your skin removed)",
        "Heartbreak-induced thermal inflammation (go sleep with wilson)",
    ],
    "lump": [
        "Moron gland hyperplasia",
        "Spontaneous stupid tumor",
        "pseudonodulic IQ-deficiency growth",
        "Metaphysical Meatball disease. (We need to remove your skin)",
        "Psychosomatic flesh bundle (you're high go home)",
        "Moron-induced granuloma (You have 6 months to live)",
    ],
    "idiocy": [
        "Cerebral decelerosis (your thoughts run Windows 95)",
        "Acute moronitis with chronic backpedaling",
        "Smugnosia terminalis (inability to shut up despite zero knowledge)",
        "Reality rejection flare-up (triggered by facts)",
        "Cortical slowdown with live-commentary syndrome",
        "Neurohumoral collapse caused by confidence in conspiracy theories",
        "Post-truth lobular erosion",
        "Flatline of insight with TikTok overlays",
        "Meme-induced logical infarct",
        "Memory hole echo chamber disorder",
        "Dunning-Krugerosis, advanced stage",
        "Terminal intellect disorder",
        "Logical non-compliance syndrome",
        "Hippocampal inflammation due to Reddit overdose",
        "Recurring psychosomatic derp",
        "Cortical smug overload",
        "Idiopathic synthetic glandular hyperplasia (you literally have no brain cells)",
        "You have a cold.",
    ],
}

# ------------------------------
# Lazy, threadsafe model loader (so cold starts don't blank-screen)
# ------------------------------
_model = None
_model_lock = threading.Lock()
_preprocess = None
_decode = None
_load_img = None
_img_to_array = None

def _ensure_model():
    global _model, _preprocess, _decode, _load_img, _img_to_array
    if _model is not None:
        return
    with _model_lock:
        if _model is not None:
            return
        log.info("Loading MobileNetV2 (imagenet) ...")
        # Import inside to avoid heavy import at module import time
        import tensorflow as tf  # noqa: F401
        from tensorflow.keras.applications.mobilenet_v2 import (
            MobileNetV2,
            preprocess_input,
            decode_predictions,
        )
        from tensorflow.keras.preprocessing import image as kimage
        _model = MobileNetV2(weights="imagenet")
        _preprocess = preprocess_input
        _decode = decode_predictions
        _load_img = kimage.load_img
        _img_to_array = kimage.img_to_array
        log.info("Model loaded.")

# Optional: manual warm route to pre-load the model
_warm_started = False
def _start_warmup_once():
    global _warm_started
    if not _warm_started:
        _warm_started = True
        threading.Thread(target=_ensure_model, daemon=True).start()

@app.get("/__warmup")
def warmup():
    _start_warmup_once()
    return {"warming": True}

# ------------------------------
# Helpers
# ------------------------------
ALLOWED_EXTS = {"png", "jpg", "jpeg", "webp"}

def _allowed(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTS

def classify_image(img_path: str) -> str:
    try:
        _ensure_model()
        img = _load_img(img_path, target_size=(224, 224))
        x = _img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = _preprocess(x)
        preds = _model.predict(x)
        label = _decode(preds, top=1)[0][0][1]
        return label.lower()
    except Exception as e:
        log.warning("Classification error: %s", e)
        log.debug("\n" + traceback.format_exc())
        return "idiocy"

def map_to_tag(label: str) -> str:
    try:
        if random.random() < 0.07:
            return "idiocy"
        if any(w in label for w in ["rash", "eczema", "blister", "hives"]):
            return "rash"
        if any(w in label for w in ["bruise", "contusion", "hematoma"]):
            return "bruise"
        if any(w in label for w in ["swelling", "edema", "inflammation"]):
            return "swelling"
        if any(w in label for w in ["lump", "tumor", "mass", "growth"]):
            return "growth"
        if any(w in label for w in ["burn", "scald", "char"]):
            return "burn"
        return random.choice(list(diagnosis_bank.keys()))
    except Exception:
        return "idiocy"

QUOTES = [
    "Everybody lies.",
    "It's never lupus.",
    "Patients always lie, especially to doctors.",
    "If you talk to God, you're religious. If God talks to you, you're psychotic.",
    "I take risks, sometimes patients die. But not taking risks causes more deaths.",
]

def generate_diagnosis(img_path: Optional[str]) -> tuple[str, str]:
    try:
        label = classify_image(img_path) if img_path else "idiocy"
        tag = map_to_tag(label)
        return random.choice(diagnosis_bank[tag]), random.choice(QUOTES)
    except Exception as e:
        log.error("Diagnosis generation error: %s", e)
        log.debug("\n" + traceback.format_exc())
        return "Diagnostic error. Likely idiocy.", random.choice(QUOTES)

# ------------------------------
# Routes
# ------------------------------
@app.get("/")
def home_get():
    _start_warmup_once()  # kick off background model load on first visit
    return render_template("index.html")

@app.post("/")
@app.post("/diagnose")
def home_post():
    if "image" not in request.files:
        abort(400, "image field missing")
    f = request.files["image"]
    if not f.filename:
        abort(400, "no file selected")
    if not _allowed(f.filename):
        abort(400, "unsupported file type")

    fname = secure_filename(f.filename)
    path = os.path.join(app.config["UPLOAD_FOLDER"], fname)

    # Process entirely in-memory first to avoid partial writes
    try:
        raw = f.read()
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        img.thumbnail((1024, 1024))
        img.save(path, format="JPEG", quality=90)
    except Exception as e:
        log.exception("Failed to read/save image: %s", e)
        abort(400, "invalid image")

    dx, quote = generate_diagnosis(path)
    return render_template("index.html", diagnosis=dx, random_quote=quote)

@app.get("/__health")
def health():
    return {"ok": True}

@app.errorhandler(400)
def bad_request(e):
    return render_template("error.html", message=str(e)), 400

@app.errorhandler(500)
def server_error(e):
    log.exception("500 error: %s", e)
    return render_template("error.html", message="Internal error. The ducklings are investigating."), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5001)), debug=True)
