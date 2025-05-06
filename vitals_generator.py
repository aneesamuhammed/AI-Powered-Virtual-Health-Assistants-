import random
import json
from datetime import datetime

# Function to generate realistic heart rate and blood pressure
def generate_vitals():
    # Generate Heart Rate (normal: 60-100 bpm)
    heart_rate = random.randint(60, 100)

    # Generate Blood Pressure (systolic/diastolic, normal: 90/60 - 120/80 mmHg)
    systolic_bp = random.randint(90, 120)
    diastolic_bp = random.randint(60, 80)

    # Simulate activity (Rest, Walk, Run)
    activity = random.choice(["Rest", "Walking", "Running"])

    # Adjust vitals based on activity
    if activity == "Running":
        heart_rate += random.randint(20, 50)
    elif activity == "Walking":
        heart_rate += random.randint(5, 15)

    # Return only necessary vitals
    return {
        "heart_rate": heart_rate,
        "systolic_bp": systolic_bp,
        "diastolic_bp": diastolic_bp
    }

# API-Ready Function to Get Vitals
def get_vitals():
    return json.dumps(generate_vitals())

# Test the function
if __name__ == "__main__":
    print(get_vitals())
