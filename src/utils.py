import os 
import sys 
import numpy as np
import pandas as pd
import dill
import pickle

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import Custom_Exception
from src.logger_config import logger

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise Custom_Exception(e,sys)

from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

def eval_model(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}
        fitted_models = {}

        for model_name, model in models.items():
            param_grid = params.get(model_name, {})

            gs = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, verbose=0)
            gs.fit(X_train, y_train)

            best_model = gs.best_estimator_

            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            train_score = accuracy_score(y_train, y_train_pred)
            test_score = accuracy_score(y_test, y_test_pred)

            report[model_name] = test_score
            fitted_models[model_name] = best_model  # ✅ Save fitted model

        return report, fitted_models  # ✅ Return both
    except Exception as e:
        raise Custom_Exception(e, sys)

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise Custom_Exception(e, sys)