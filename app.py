import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow import keras
from PIL import Image
import io
import base64

# NumPy np.object 패치 (오류 방지)
try:
    np.object_
except AttributeError:
    np.object_ = object

app = Flask(__name__)

# 모델 로드
model = keras.models.load_model('model_final_v1_5.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        image_data = base64.b64decode(request.json['image'].split(',')[1])
        image = Image.open(io.BytesIO(image_data))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = image.resize((32, 32))
        image_array = np.array(image, dtype=np.float32)
        image_array = np.expand_dims(image_array, axis=0)
        
        prediction = model.predict(image_array)[0][0]
        confidence = float(prediction)
        
        result = "REAL" if confidence < 0.5 else "FAKE"
        result_confidence = confidence if result == "FAKE" else (1-confidence)
        
        return jsonify({
            'result': result,
            'confidence': f"{result_confidence:.2%}",
            'raw_score': confidence
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
