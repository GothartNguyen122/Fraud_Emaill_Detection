from flask import Flask, request, jsonify
import joblib
from bs4 import BeautifulSoup

app = Flask(__name__)

#Load Model
model_path = 'fraud_email_model.pkl'
model = joblib.load(model_path)

#Pre_Processing Data
def preprocess_text(text):
    # Remove HTML Tagss
    text = BeautifulSoup(text, "html.parser").text
    return text

@app.route('/predict', methods=['POST'])
def predict():
    try:
        #Get Data from Front_End
        data = request.get_json()
        text = data.get("text", "")
        
        if not text:
            return jsonify({"error": "No text provided"}), 400

        #Execute Pre_Processing Data
        text = preprocess_text(text)

        #Predicting 
        prediction = model.predict([text])[0]

        #Return Result 
        return jsonify({"prediction": int(prediction)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
