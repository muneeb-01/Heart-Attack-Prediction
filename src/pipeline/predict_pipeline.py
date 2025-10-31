import os
import sys
import pandas as pd
from src.exception import Custom_Exception
from src.utils import load_object
from src.logger_config import logger


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        """
        Predicts heart disease using the trained model and preprocessor.
        """
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

            logger.info("🔹 Loading model and preprocessor artifacts...")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            logger.info("✅ Model and preprocessor successfully loaded.")

            logger.info("🔹 Transforming input features...")
            data_scaled = preprocessor.transform(features)

            logger.info("🔹 Generating predictions...")
            preds = model.predict(data_scaled)

            if hasattr(model, "predict_proba"):
                probability = model.predict_proba(data_scaled)[0][1] * 100
            else:
                probability = None  # Some models don’t support this
                
            logger.info("✅ Prediction completed successfully.")

            return (preds, probability)

        except Exception as e:
            logger.error(f"❌ Error during prediction: {e}")
            raise Custom_Exception(e, sys)


class CustomData:
    def __init__(
        self,
        Age: int,
        RestingBP: int,
        Cholesterol: int,
        FastingBS: int,
        MaxHR: int,
        Oldpeak: float,
        Sex_F: int,
        Sex_M: int,
        ChestPainType_ASY: int,
        ChestPainType_ATA: int,
        ChestPainType_NAP: int,
        ChestPainType_TA: int,
        RestingECG_LVH: int,
        RestingECG_Normal: int,
        RestingECG_ST: int,
        ExerciseAngina_N: int,
        ExerciseAngina_Y: int,
        ST_Slope_Down: int,
        ST_Slope_Flat: int,
        ST_Slope_Up: int,
    ):
        """
        Custom data structure for single prediction input.
        """
        self.Age = Age
        self.RestingBP = RestingBP
        self.Cholesterol = Cholesterol
        self.FastingBS = FastingBS
        self.MaxHR = MaxHR
        self.Oldpeak = Oldpeak
        self.Sex_F = Sex_F
        self.Sex_M = Sex_M
        self.ChestPainType_ASY = ChestPainType_ASY
        self.ChestPainType_ATA = ChestPainType_ATA
        self.ChestPainType_NAP = ChestPainType_NAP
        self.ChestPainType_TA = ChestPainType_TA
        self.RestingECG_LVH = RestingECG_LVH
        self.RestingECG_Normal = RestingECG_Normal
        self.RestingECG_ST = RestingECG_ST
        self.ExerciseAngina_N = ExerciseAngina_N
        self.ExerciseAngina_Y = ExerciseAngina_Y
        self.ST_Slope_Down = ST_Slope_Down
        self.ST_Slope_Flat = ST_Slope_Flat
        self.ST_Slope_Up = ST_Slope_Up

    def get_data_as_data_frame(self):
        """
        Converts input data into a pandas DataFrame for model processing.
        """
        try:
            logger.info("🔹 Converting custom input data into DataFrame...")

            custom_data_input_dict = {
                "Age": [self.Age],
                "RestingBP": [self.RestingBP],
                "Cholesterol": [self.Cholesterol],
                "FastingBS": [self.FastingBS],
                "MaxHR": [self.MaxHR],
                "Oldpeak": [self.Oldpeak],
                "Sex_F": [self.Sex_F],
                "Sex_M": [self.Sex_M],
                "ChestPainType_ASY": [self.ChestPainType_ASY],
                "ChestPainType_ATA": [self.ChestPainType_ATA],
                "ChestPainType_NAP": [self.ChestPainType_NAP],
                "ChestPainType_TA": [self.ChestPainType_TA],
                "RestingECG_LVH": [self.RestingECG_LVH],
                "RestingECG_Normal": [self.RestingECG_Normal],
                "RestingECG_ST": [self.RestingECG_ST],
                "ExerciseAngina_N": [self.ExerciseAngina_N],
                "ExerciseAngina_Y": [self.ExerciseAngina_Y],
                "ST_Slope_Down": [self.ST_Slope_Down],
                "ST_Slope_Flat": [self.ST_Slope_Flat],
                "ST_Slope_Up": [self.ST_Slope_Up],
            }

            df = pd.DataFrame(custom_data_input_dict)
            logger.info(f"✅ DataFrame created with shape: {df.shape}")
            return df

        except Exception as e:
            logger.error(f"❌ Error creating DataFrame: {e}")
            raise Custom_Exception(e, sys)
