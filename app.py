from flask import Flask, render_template, request
from io import BytesIO
from PIL import Image
import os

app = Flask(__name__)

# Allow up to 20 MB uploads
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024

# Simple home page (GET) and upload handler (POST)
@app.route("/", methods=["GET", "POST"])
def home():
    diagnosis = None
    random_quote = None

    if request.method == "POST":
        f = request.files.get("image")
        if not f or f.filename == "":
            diagnosis = None
            random_quote = None
            return render_template("index.html", diagnosis=diagnosis, random_quote=random_quote, error="No file uploaded")

        # Shrink big phone photos so processing is fast + avoids 502s
        try:
            img = Image.open(f.stream).convert("RGB")
            img.thumbnail((1600, 1600))  # preserves aspect ratio

            buf = BytesIO()
            img.save(buf, format="JPEG", quality=85, optimize=True)
            buf.seek(0)

            # TODO: Replace this mock with your real diagnosis call,
            #       using 'buf.getvalue()' as the image bytes.
            diagnosis = "Temporary: server healthy and upload works."
            random_quote = "“Everybody lies.”"

        except Exception as e:
            return render_template("index.html", diagnosis=None, random_quote=None, error=f"Server error: {e}"), 500

    return render_template("index.html", diagnosis=diagnosis, random_quote=random_quote)

if __name__ == "__main__":
    # Local dev
    app.run(host="0.0.0.0", port=5000, debug=True)
