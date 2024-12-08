from flask import Flask, request, jsonify
import joblib

# Load mô hình và vectorizer
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '')
    # Vector hóa văn bản
    text_vectorized = vectorizer.transform([text])
    # Dự đoán
    prediction = model.predict(text_vectorized)[0]
    return jsonify({'prediction': int(prediction)})

if __name__ == '__main__':
    app.run(debug=True)
