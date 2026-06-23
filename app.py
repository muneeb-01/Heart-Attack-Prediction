from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application


# ✅ Home route
@app.route('/')
def index():
    return render_template('index.html')


# ✅ Prediction route
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')

    else:
        try:
            # 1. Parse and validate numerical fields
            age_str = request.form.get('Age')
            resting_bp_str = request.form.get('RestingBP')
            cholesterol_str = request.form.get('Cholesterol')
            fasting_bs_str = request.form.get('FastingBS')
            max_hr_str = request.form.get('MaxHR')
            oldpeak_str = request.form.get('Oldpeak')

            errors = []
            
            # Check for empty numerical fields
            if not all([age_str, resting_bp_str, cholesterol_str, fasting_bs_str, max_hr_str, oldpeak_str]):
                errors.append("All health metrics (numerical fields) are required.")
                return render_template('home.html', errors=errors, form_data=request.form)

            try:
                age = int(age_str)
                resting_bp = int(resting_bp_str)
                cholesterol = int(cholesterol_str)
                fasting_bs = int(fasting_bs_str)
                max_hr = int(max_hr_str)
                oldpeak = float(oldpeak_str)
            except ValueError:
                errors.append("Please enter valid numeric values for health metrics.")
                return render_template('home.html', errors=errors, form_data=request.form)

            # 2. Extract categorical fields
            sex = request.form.get('sex_option')
            chest_pain = request.form.get('chestpain_option')
            resting_ecg = request.form.get('restingecg_option')
            exercise_angina = request.form.get('exerciseangina_option')
            st_slope = request.form.get('stslope_option')

            # 3. Clinical range and value checks
            if not (1 <= age <= 120):
                errors.append("Age must be between 1 and 120.")
            if not (50 <= resting_bp <= 250):
                errors.append("Resting Blood Pressure must be between 50 and 250 mm Hg.")
            if not (0 <= cholesterol <= 600):
                errors.append("Cholesterol must be between 0 and 600 mg/dl.")
            if fasting_bs not in [0, 1]:
                errors.append("Fasting Blood Sugar must be either 'Below 120 mg/dl' or 'Above 120 mg/dl'.")
            if not (60 <= max_hr <= 220):
                errors.append("Max Heart Rate must be between 60 and 220 bpm.")
            if not (-2.0 <= oldpeak <= 10.0):
                errors.append("Oldpeak (ST depression) must be between -2.0 and 10.0.")

            # Categorical validity check
            if sex not in ['M', 'F']:
                errors.append("Please select a valid option for Sex.")
            if chest_pain not in ['ASY', 'ATA', 'NAP', 'TA']:
                errors.append("Please select a valid option for Chest Pain Type.")
            if resting_ecg not in ['Normal', 'LVH', 'ST']:
                errors.append("Please select a valid option for Resting ECG Result.")
            if exercise_angina not in ['Y', 'N']:
                errors.append("Please select a valid option for Exercise Induced Angina.")
            if st_slope not in ['Up', 'Flat', 'Down']:
                errors.append("Please select a valid option for ST Slope.")

            if errors:
                return render_template('home.html', errors=errors, form_data=request.form)

            # 4. Correct Python-side One-Hot Encoding
            Sex_F = 1 if sex == 'F' else 0
            Sex_M = 1 if sex == 'M' else 0

            ChestPainType_ASY = 1 if chest_pain == 'ASY' else 0
            ChestPainType_ATA = 1 if chest_pain == 'ATA' else 0
            ChestPainType_NAP = 1 if chest_pain == 'NAP' else 0
            ChestPainType_TA  = 1 if chest_pain == 'TA'  else 0

            RestingECG_LVH = 1 if resting_ecg == 'LVH' else 0
            RestingECG_Normal = 1 if resting_ecg == 'Normal' else 0
            RestingECG_ST = 1 if resting_ecg == 'ST' else 0

            ExerciseAngina_N = 1 if exercise_angina == 'N' else 0
            ExerciseAngina_Y = 1 if exercise_angina == 'Y' else 0

            ST_Slope_Down = 1 if st_slope == 'Down' else 0
            ST_Slope_Flat = 1 if st_slope == 'Flat' else 0
            ST_Slope_Up = 1 if st_slope == 'Up' else 0

            # 5. Load into CustomData structure
            data = CustomData(
                Age=age,
                RestingBP=resting_bp,
                Cholesterol=cholesterol,
                FastingBS=fasting_bs,
                MaxHR=max_hr,
                Oldpeak=oldpeak,
                Sex_F=Sex_F,
                Sex_M=Sex_M,
                ChestPainType_ASY=ChestPainType_ASY,
                ChestPainType_ATA=ChestPainType_ATA,
                ChestPainType_NAP=ChestPainType_NAP,
                ChestPainType_TA=ChestPainType_TA,
                RestingECG_LVH=RestingECG_LVH,
                RestingECG_Normal=RestingECG_Normal,
                RestingECG_ST=RestingECG_ST,
                ExerciseAngina_N=ExerciseAngina_N,
                ExerciseAngina_Y=ExerciseAngina_Y,
                ST_Slope_Down=ST_Slope_Down,
                ST_Slope_Flat=ST_Slope_Flat,
                ST_Slope_Up=ST_Slope_Up
            )

            # Convert to DataFrame
            pred_df = data.get_data_as_data_frame()

            # Predict
            predict_pipeline = PredictPipeline()
            results, probability = predict_pipeline.predict(pred_df)
            
            if results[0] == 1:
                prediction_label = "Heart Disease Detected 💔"
                is_high_risk = True
            else:
                prediction_label = "No Heart Disease ❤️"
                is_high_risk = False

            return render_template(
                'home.html', 
                results=prediction_label, 
                probability=round(probability, 2) if probability is not None else None,
                is_high_risk=is_high_risk,
                form_data=request.form
            )

        except Exception as e:
            print(f"Error during prediction: {e}")
            return render_template('home.html', errors=[f"An unexpected error occurred during prediction: {e}"], form_data=request.form)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
