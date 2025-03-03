from flask import Flask, request, jsonify
from flask_sslify import SSLify
from dotenv import load_dotenv
import os
import pickle
import numpy as np

app = Flask(__name__)
sslify = SSLify(app)  # Convert HTTP to HTTPS

# Load environment variables from .env
load_dotenv()
API_KEY = os.getenv('FLASK_API_KEY')
if not API_KEY:
    raise ValueError("La clé API n'est pas définie dans .env. Vérifiez le fichier .env.")

# Load the trained model
with open('cv_score_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/predict_cv_score', methods=['POST'])
def predict_cv_score():
    try:
        auth_key = request.headers.get('X-API-Key')
        if not auth_key or auth_key != API_KEY:
            return jsonify({'error': 'Clé API invalide'}), 401

        data = request.get_json()
        # Extract features from JSON, ensuring they are numeric
        edu_level = float(data.get('education_level', 0))
        years = float(data.get('years_of_experience', 0))
        skills_count = float(data.get('skills_count', 0))
        certs_count = float(data.get('certifications_count', 0))

        # Prepare input for model
        features = np.array([[edu_level, years, skills_count, certs_count]])
        score = model.predict(features)[0]

        # Cap score at 0-100 and ensure it's an integer
        score = min(100, max(0, int(score)))
        return jsonify({'score': score})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, ssl_context='adhoc')  # Use SSL for HTTPS