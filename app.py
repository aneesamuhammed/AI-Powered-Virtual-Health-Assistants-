from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
import pickle
from vitals_generator import generate_vitals

app = Flask(__name__)

# Load trained model and data
model = tf.keras.models.load_model("chatbot_model.hdf5")
words = pickle.load(open("words.pkl", "rb"))
labels = pickle.load(open("labels.pkl", "rb"))


def preprocess_input(symptoms):
    symptoms = symptoms.lower().split(", ")  # Convert to list
    bag = [1 if w in symptoms else 0 for w in words]
    return np.array([bag])

@app.route('/')
def home():
    return render_template('index.html')  # Serve the index.html template

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    if not data or 'symptoms' not in data:
        return jsonify({"error": "No symptoms provided"}), 400

    symptoms = data["symptoms"]
    vitals_raw = generate_vitals()  # Always generate vitals, not used in prediction

    # Convert all vitals to regular Python float
    vitals = {k: float(v) for k, v in vitals_raw.items()}

    # Preprocess symptoms
    input_data = preprocess_input(symptoms)

    # Model prediction
    predictions = model.predict(input_data)[0]

    # Get top 5 indices (adjust as needed)
    top_indices = np.argsort(predictions)[::-1][:5]
    top_predictions = [predictions[i] for i in top_indices]

    # Normalize top probabilities to sum to 100%
    total = sum(top_predictions)
    normalized = [p / total for p in top_predictions]

    results = [
        {
            "disease": labels[top_indices[i]],
            "probability": round(float(normalized[i]) * 100, 1)  # Ensure JSON serializable
        }
        for i in range(len(top_indices))
    ]

    return jsonify({"predictions": results, "vitals": vitals})

if __name__ == "__main__":
    app.run(debug=True)
