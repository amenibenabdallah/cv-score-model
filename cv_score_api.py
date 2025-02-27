from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('cv_score_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/predict_cv_score', methods=['POST'])
def predict_cv_score():
    try:
        data = request.get_json()
        # Extract features from JSON
        edu_level = float(data.get('education_level', 0))
        years = float(data.get('years_of_experience', 0))
        skills_count = float(data.get('skills_count', 0))
        certs_count = float(data.get('certifications_count', 0))

        # Prepare input for model
        features = np.array([[edu_level, years, skills_count, certs_count]])
        score = model.predict(features)[0]

        # Cap score at 0-100
        score = min(100, max(0, int(score)))
        return jsonify({'score': score})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)