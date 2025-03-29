import onnxruntime as rt
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the ONNX model
model_path = "model.onnx"
session = rt.InferenceSession(model_path)
input_name = session.get_inputs()[0].name

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Extract features from request
        features = np.array([
            data['gender'], data['height'], data['weight'], data['ap_hi'],
            data['ap_lo'], data['cholesterol'], data['gluc'], data['smoke'],
            data['alco'], data['active'], data['age_years'], data['bmi']
        ], dtype=np.float32).reshape(1, -1)
        
        # Make prediction
        prediction = session.run(None, {input_name: features})[0]
        result = float(prediction[0])
        
        return jsonify({"prediction": result})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
