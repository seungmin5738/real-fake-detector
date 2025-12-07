import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow import keras
from PIL import Image
import io
import base64

# NumPy np.object 패치 (버전 차이 오류 방지)
try:
    np.object_
except AttributeError:
    np.object_ = object

app = Flask(__name__)

# --------- 모델 로드 ---------
# 변환한 모델 파일을 쓰는 경우: "model_final_v1_5_render.h5"
MODEL_PATH = "model_final_v1_5_render.h5"  # 필요하면 model_final_v1_5.h5 로 바꿔도 됨

model = keras.models.load_model(
    MODEL_PATH,
    compile=False,
    safe_mode=False  # 구버전 Keras로 저장된 모델 로드할 때 에러 줄이는 옵션
)

# --------- 라우트 ---------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. 프론트에서 넘어온 base64 이미지 읽기
        image_b64 = request.json.get('image', None)
        if image_b64 is None:
            return jsonify({'error': 'No image data provided'}), 400

        # "image/xxx;base64,...." 형식이므로 콤마 뒤만 사용
        image_data = base64.b64decode(image_b64.split(',')[1])
        image = Image.open(io.BytesIO(image_data))

        # 2. 전처리: RGB 변환, 32x32 리사이즈 (정규화 X)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = image.resize((32, 32))
        image_array = np.array(image, dtype=np.float32)
        image_array = np.expand_dims(image_array, axis=0)  # (1, 32, 32, 3)

        # 3. 예측
        pred = model.predict(image_array)[0][0]  # 0~1 사이 스칼라라고 가정
        score = float(pred)

        # 0.5 기준으로 REAL / FAKE 판단 (원래 학습 로직에 맞게 조절 가능)
        label = "REAL" if score < 0.5 else "FAKE"
        confidence = score if label == "FAKE" else (1 - score)

        return jsonify({
            'result': label,
            'confidence': f"{confidence:.2%}",
            'raw_score': score
        })

    except Exception as e:
        # 어떤 에러든 프론트에서 볼 수 있게 반환
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # 로컬 개발용
    app.run(host='0.0.0.0', port=5000, debug=True)
