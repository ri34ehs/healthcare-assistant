from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib

app = Flask(__name__)
CORS(app)

# Load trained model
model = joblib.load('diabetes_model.pkl')

def get_diet_plan_by_age(age):
    if age < 18:
        return {
            "Breakfast": ["Whole grain cereal with milk", "Fruit smoothie"],
            "Lunch": ["Grilled chicken wrap", "Carrot sticks"],
            "Dinner": ["Baked fish", "Brown rice", "Steamed veggies"],
            "Snacks": ["Apple slices", "Yogurt"]
        }
    elif 18 <= age <= 40:
        return {
            "Breakfast": ["Oatmeal with berries", "Boiled eggs"],
            "Lunch": ["Quinoa salad", "Grilled paneer or chicken"],
            "Dinner": ["Vegetable soup", "Grilled fish/chicken"],
            "Snacks": ["Nuts", "Low-fat Greek yogurt"]
        }
    elif 41 <= age <= 60:
        return {
            "Breakfast": ["Multigrain toast", "Low-sugar fruits"],
            "Lunch": ["Steamed vegetables", "Brown rice and dal"],
            "Dinner": ["Lentil soup", "Salad with olive oil"],
            "Snacks": ["Roasted chana", "Cucumber sticks"]
        }
    else:
        return {
            "Breakfast": ["Soft boiled egg", "Whole wheat porridge"],
            "Lunch": ["Khichdi with vegetables", "Buttermilk"],
            "Dinner": ["Mashed vegetables", "Light soup"],
            "Snacks": ["Banana", "Low-sugar biscuits"]
        }

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        features = np.array([
            float(data.get('Pregnancies', 0)),
            float(data.get('Glucose', 0)),
            float(data.get('BloodPressure', 0)),
            float(data.get('SkinThickness', 0)),
            float(data.get('Insulin', 0)),
            float(data.get('BMI', 0)),
            float(data.get('DiabetesPedigreeFunction', 0)),
            float(data.get('Age', 0))
        ]).reshape(1, -1)

        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]

        # Risk level based on probability
        if probability < 0.3:
            risk_level = "Low Risk"
        elif probability < 0.7:
            risk_level = "Moderate Risk"
        else:
            risk_level = "High Risk"

        age = int(data.get('Age', 0))
        diet_plan = get_diet_plan_by_age(age)

        bmi_val = float(data.get('BMI', 0))
        if bmi_val < 18.5:
            bmi_status = "Underweight"
        elif bmi_val < 25:
            bmi_status = "Normal weight"
        elif bmi_val < 30:
            bmi_status = "Overweight"
        else:
            bmi_status = "Obese"

        if age < 18:
            calories = '2000 - 2200 kcal/day'
            water = '1.5 - 2 liters/day'
        elif age <= 40:
            calories = '2200 - 2500 kcal/day'
            water = '2 - 2.5 liters/day'
        elif age <= 60:
            calories = '2000 - 2200 kcal/day'
            water = '1.8 - 2.2 liters/day'
        else:
            calories = '1800 - 2000 kcal/day'
            water = '1.5 - 2 liters/day'

        return jsonify({
            'prediction': int(prediction),
            'riskLevel': risk_level,
            'probability': round(probability, 2),
            'dietPlan': diet_plan,
            'bmiStatus': bmi_status,
            'recommendedCalories': calories,
            'recommendedWater': water
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get("message", "").lower()

    if any(keyword in user_message for keyword in ["bmi", "body mass index"]):
        return jsonify({"response": "To calculate BMI, please enter your height and weight."})
    elif any(keyword in user_message for keyword in ["diabetes", "predict", "blood sugar"]):
        return jsonify({"response": "Please fill the medical form to predict diabetes."})
    elif any(greet in user_message for greet in ["hello", "hi", "hey", "good morning", "good evening"]):
        return jsonify({"response": "Hello! I'm your AI Health Assistant. How can I help?"})
    elif any(keyword in user_message for keyword in ["diet", "nutrition", "food", "meal", "healthy eating"]):
        return jsonify({"response": "I can suggest diet plans based on your age and health. Please provide your age."})
    elif any(keyword in user_message for keyword in ["help", "how", "use", "instructions", "guide"]):
        return jsonify({"response": "You can ask me to predict diabetes, calculate BMI, or suggest diet plans. Just type your query."})
    else:
        return jsonify({"response": "I'm still learning! Try asking about diabetes, BMI, diet, or how to use the app."})

if __name__ == '__main__':
    app.run(debug=True)
