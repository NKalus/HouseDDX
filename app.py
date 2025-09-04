import os
import random
import importlib
from io import BytesIO

from flask import Flask, render_template, request, redirect, url_for, flash
from PIL import Image

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "house-always-right")

# ---------- Optional TensorFlow ----------
USE_TF = importlib.util.find_spec("tensorflow") is not None
tf = None
model = None

def ensure_model():
    """Load TF model only if TensorFlow is installed; otherwise stay jokes-only."""
    global tf, model
    if not USE_TF:
        return
    if model is None:
        import tensorflow as _tf  # noqa: E402
        from tensorflow.keras.applications.mobilenet_v2 import (
            MobileNetV2, preprocess_input, decode_predictions
        )  # noqa: E402
        # Keep references so we can call later
        globals()["tf"] = _tf
        globals()["preprocess_input"] = preprocess_input
        globals()["decode_predictions"] = decode_predictions
        globals()["MobileNetV2"] = MobileNetV2

        model = MobileNetV2(weights="imagenet")

# ---------- Your House-style diagnoses & quotes ----------
JOKE_DIAGNOSES = [
    "Spontaneous stupid tumor",
    "Vesiculopathic keratinocyte rebellion (your skin cells are evil)",
    "Consequential thermogenic moron disease",
    "Cortical smug overload",
    "Logical non-compliance syndrome",
    "Pseudolymphatic egomania (your ego is retaining water)",
    "Overconfidence blob (resistant to feedback)",
]

HOUSE_QUOTES = [
    "“It’s never lupus.”",
    "“Everybody lies.”",
    "“Patients always lie, especially to doctors.”",
    "“I take risks, sometimes patients die. But not taking risks causes more deaths.”",
]

ALLOWED_EXTS = {"png", "jpg", "jpeg", "webp"}


def allowed(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTS


@app.route("/", methods=["GET", "POST"])
def index():
    diagnosis = None
    quote = None

    if request.method == "POST":
        file = request.files.get("image")
        if not file or file.filename == "":
            flash("Please choose an image.")
            return redirect(url_for("index"))

        if not allowed(file.filename):
            flash("Only PNG/JPG/JPEG/WEBP images are supported.")
            return redirect(url_for("index"))

        # Read image bytes safely; PIL validates file
        try:
            img_bytes = BytesIO(file.read())
            _ = Image.open(img_bytes).convert("RGB")
        except Exception:
            flash("Could not read that image. Try another file.")
            return redirect(url_for("index"))

        # If TF exists and you ever want to use it, you can switch here.
        if USE_TF:
            # Minimal demo: run ImageNet & turn top-1 into a mock 'dx'
            ensure_model()
            if model is not None:
                import numpy as np
                img = Image.open(img_bytes).convert("RGB").resize((224, 224))
                arr = np.array(img)[None, ...].astype("float32")
                arr = preprocess_input(arr)
                preds = model.predict(arr)
                decoded = decode_predictions(preds, top=1)[0][0]
                diagnosis = f"ImageNet says: {decoded[1]} ({decoded[2]*100:.1f}% sure). House disagrees."
            else:
                # Shouldn't happen, but fall back just in case.
                diagnosis = random.choice(JOKE_DIAGNOSES)
        else:
            # Jokes-only mode
            diagnosis = random.choice(JOKE_DIAGNOSES)

        quote = random.choice(HOUSE_QUOTES)

    return render_template("index.html", diagnosis=diagnosis, random_quote=quote)


if __name__ == "__main__":
    # Local run
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
