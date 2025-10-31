import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import Custom_Exception
from src.logger_config import logger
import os

from src.utils import save_object



@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")



class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    
    def get_data_transformer_object(self):
        try:
            numerical_columns = [ "Age","RestingBP","Cholesterol","FastingBS","MaxHR","Oldpeak"]
            categorical_columns = [ "Sex_F", "Sex_M","ChestPainType_ASY", "ChestPainType_ATA", "ChestPainType_NAP", "ChestPainType_TA",
                                    "RestingECG_LVH", "RestingECG_Normal", "RestingECG_ST",
                                    "ExerciseAngina_N", "ExerciseAngina_Y",
                                    "ST_Slope_Down", "ST_Slope_Flat", "ST_Slope_Up"
                                ]
            
            logger.info("Categorical and Numerical features separated.")
            logger.info(f"Categorical columns: {categorical_columns}")
            logger.info(f"Numerical columns: {numerical_columns}")
            
            num_pipeline= Pipeline(
                    steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                    ]
                )
            cat_pipeline=Pipeline(
                    steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))
                    ]
                )
            preprocessor=ColumnTransformer(
                    [
                    ("num_pipeline",num_pipeline,numerical_columns),
                    ("cat_pipelines",cat_pipeline,categorical_columns)
                    ]
                )
            logger.info("Preprocessor created successfully.")
            return preprocessor
        except Exception as e:
            raise Custom_Exception(e,sys)

    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logger.info("Read train and test data completed")
            
            preprocessing_obj = self.get_data_transformer_object()
            logger.info("Preprocessor Obtained.")
            
            target_column_name = "HeartDisease"
            
            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]
            
            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]
            
            logger.info("Input & target features (train & test) separated from dataframe.")
            
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)
            
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            logger.info("Preprocessing Object saved.")
            return (train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path)
        except Exception as e:
            raise Custom_Exception(e,sys)