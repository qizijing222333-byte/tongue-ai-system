import json
import numpy as np
import cv2
from flask import Flask, request, jsonify, render_template

from inference.quality_gate import load_config
from inference.pipeline import run_pipeline

app = Flask(__name__)

CFG = load_config()
app.config["MAX_CONTENT_LENGTH"] = 6 * 1024 * 1024  # 6MB


@app.get("/health")
def health():
    return jsonify({"ok": True})

@app.get("/")
def index():
    return render_template("index.html", result=None, raw_json=None, image_preview=None)

@app.post("/api/predict")
def predict():
    if "image" not in request.files:
        return jsonify({"error": "missing form-data field: image"}), 400

@app.post("/web/predict")
def web_predict():
    if "image" not in request.files:
        return render_template("index.html", result=None, raw_json="missing image", image_preview=None), 400

    f = request.files["image"]
    data = f.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return render_template("index.html", result=None, raw_json="cannot decode image", image_preview=None), 400

    return_debug = request.form.get("return_debug", "false").lower() in ("1", "true", "yes", "on")

    out = run_pipeline(img, CFG, return_debug=return_debug)
    raw_json = json.dumps(out, ensure_ascii=False, indent=2)

    # 简单图片预览：用 data URL（避免存文件）
    import base64
    b64 = base64.b64encode(data).decode("utf-8")
    mime = f.mimetype or "image/jpeg"
    image_preview = f"data:{mime};base64,{b64}"

    return render_template("index.html", result=out, raw_json=raw_json, image_preview=image_preview)

    f = request.files["image"]
    data = f.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "cannot decode image, please upload jpg/png"}), 400

    return_debug = request.form.get("return_debug", "false").lower() in ("1", "true", "yes")

    out = run_pipeline(img, CFG, return_debug=return_debug)

    return app.response_class(
        response=json.dumps(out, ensure_ascii=False, indent=2),
        status=200,
        mimetype="application/json"
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
