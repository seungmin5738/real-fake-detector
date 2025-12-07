import os
import io
import base64
import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow import keras
from PIL import Image

# NumPy np.object 패치 (버전 차이 오류 방지)
try:
    np.object_
except AttributeError:
    np.object_ = object

app = Flask(__name__)

# ----- 모델 로드 (절대 경로 사용) -----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model_final_v1_5_render.h5")

print("MODEL_PATH =", MODEL_PATH)
print("MODEL_EXISTS =", os.path.exists(MODEL_PATH))

model = keras.models.load_model(
    MODEL_PATH,
    compile=False,
    safe_mode=False
)

# ----- 라우트 -----
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data or "image" not in 
            return jsonify({"error": "No image data provided"}), 400

        image_b64 = data["image"]
        # "image/xxx;base64,...." 형식이므로 콤마 뒤만 사용
        img_bytes = base64.b64decode(image_b64.split(",")[1])
        image = Image.open(io.BytesIO(img_bytes))

        # 전처리: RGB 변환, 32x32 리사이즈 (정규화 X)
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = image.resize((32, 32))
        arr = np.array(image, dtype=np.float32)
        arr = np.expand_dims(arr, axis=0)  # (1, 32, 32, 3)

        # 예측 (0~1 사이 값 가정)
        pred = model.predict(arr)[0][0]
        score = float(pred)

        label = "REAL" if score < 0.5 else "FAKE"
        confidence = score if label == "FAKE" else (1 - score)

        return jsonify(
            {
                "result": label,
                "confidence": f"{confidence:.2%}",
                "raw_score": score,
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
