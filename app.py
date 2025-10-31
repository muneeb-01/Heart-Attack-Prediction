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
            # Read categorical values as single string from form
            sex = request.form.get('Sex')  # 'M' or 'F'
            chest_pain = request.form.get('ChestPainType')  # 'ASY', 'ATA', 'NAP', 'TA'
            resting_ecg = request.form.get('RestingECG')  # 'LVH', 'Normal', 'ST'
            exercise_angina = request.form.get('ExerciseAngina')  # 'Y' or 'N'
            st_slope = request.form.get('ST_Slope')  # 'Down', 'Flat', 'Up'

            # One-hot encode them manually
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

            # Now collect numerical features
            data = CustomData(
                Age=int(request.form.get('Age')),
                RestingBP=int(request.form.get('RestingBP')),
                Cholesterol=int(request.form.get('Cholesterol')),
                FastingBS=int(request.form.get('FastingBS')),
                MaxHR=int(request.form.get('MaxHR')),
                Oldpeak=float(request.form.get('Oldpeak')),
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
                prediction_label = f"Heart Disease Detected 💔 Probability: {probability:.2f}%"
            else:
                prediction_label = f"No Heart Disease ❤️ Probability: {probability:.2f}%"

            return render_template('home.html', results=prediction_label)

        except Exception as e:
            print(f"Error during prediction: {e}")
            return render_template('home.html', results=f"Error: {e}")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
