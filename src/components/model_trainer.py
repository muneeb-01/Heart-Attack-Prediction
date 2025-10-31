import sys
import os
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import r2_score, accuracy_score

from src.exception import Custom_Exception
from src.logger_config import logger
from src.utils import save_object, eval_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logger.info("Splitting up training and test input data.")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            models = {
                "Logistic Regression": LogisticRegression(),
                "KNN Classifier": KNeighborsClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
                "AdaBoost Classifier": AdaBoostClassifier(),
            }

            params = {
                "Logistic Regression": {
                    "penalty": ['l2'],
                    "C": [0.01, 0.1, 1, 10],
                    "solver": ['lbfgs', 'liblinear']
                },
                "KNN Classifier": {
                    "n_neighbors": [3, 5, 7, 9],
                    "weights": ['uniform', 'distance'],
                    "metric": ['minkowski', 'euclidean', 'manhattan']
                },
                "Decision Tree": {
                    "criterion": ['gini', 'entropy'],
                    "max_depth": [3, 5, 7, 10, None],
                    "min_samples_split": [2, 5, 10]
                },
                "Random Forest": {
                    "n_estimators": [50, 100, 200],
                    "criterion": ['gini', 'entropy'],
                    "max_depth": [5, 10, None]
                },
                "AdaBoost Classifier": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.1, 1]
                }
            }

            model_report, fitted_models = eval_model(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                params=params,
            )

            logger.info("Models Report Received.")

            best_model_score = max(model_report.values())
            best_model_name = max(model_report, key=model_report.get)

            logger.info(f"Best Model is: {best_model_name}")

            best_model = fitted_models[best_model_name]

            if best_model_score < 0.65:
                raise Custom_Exception("No Best Model found.", sys)

            logger.info(
                f"Found Best model for the training and test data: {best_model_name}"
            )

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )

            predicted = best_model.predict(X_test)
            acc = accuracy_score(y_test, predicted)

            return best_model_name, acc * 100

        except Exception as e:
            raise Custom_Exception(e, sys)
