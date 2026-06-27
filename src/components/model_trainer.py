import os
import sys
import numpy as np
from dataclasses import dataclass

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, KFold
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and testing input data arrays")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            # 1. Define our candidate model architectures
            models = {
                "Random Forest": RandomForestRegressor(random_state=42),
                "Gradient Boosting": GradientBoostingRegressor(random_state=42),
                "XGBoost": XGBRegressor(random_state=42),
                "Linear Regression": LinearRegression()
            }

            # 2. Define the hyperparameter grid spaces to explore
            params = {
                "Random Forest": {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [5, 10, None],
                    "min_samples_split": [2, 5]
                },
                "Gradient Boosting": {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.05, 0.1],
                    "max_depth": [3, 5]
                },
                "XGBoost": {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.05, 0.1],
                    "max_depth": [3, 5]
                },
                "Linear Regression": {}  # Base analytical baseline
            }

            # Setup robust 5-Fold Cross-Validation strategy
            cv_strategy = KFold(n_splits=5, shuffle=True, random_state=42)
            
            model_report: dict = {}

            # 3. Execute the Cross-Validation Hyperparameter Search Loop
            for model_name, model in models.items():
                model_param = params[model_name]
                
                logging.info(f"Starting Cross-Validation grid search tuning for: {model_name}")
                
                # Setup GridSearch with our 5-fold CV split configuration
                grid_search = GridSearchCV(
                    estimator=model,
                    param_grid=model_param,
                    cv=cv_strategy,
                    scoring='r2',
                    n_jobs=-1
                )
                
                grid_search.fit(X_train, y_train)

                # Extract the best cross-validated model instance parameters
                best_model = grid_search.best_estimator_
                
                # Evaluate generalization capability on completely unseen test data
                y_test_pred = best_model.predict(X_test)
                test_model_score = r2_score(y_test, y_test_pred)

                model_report[model_name] = (best_model, test_model_score)
                logging.info(f"Finished {model_name}. Best CV Score achieved on testing holdout: {test_model_score:.4f}")

            # 4. Isolate the absolute best performing model configuration
            best_model_score = -1
            best_model_name = ""
            best_model_obj = None

            for m_name, (m_obj, m_score) in model_report.items():
                if m_score > best_model_score:
                    best_model_score = m_score
                    best_model_name = m_name
                    best_model_obj = m_obj

            if best_model_score < 0.60:
                raise CustomException("No candidate model configuration passed the baseline architectural threshold.")

            logging.info(f"Absolute champion model discovered: {best_model_name} with R2 score: {best_model_score:.4f}")

            # 5. Freeze and serialize the tuned winning model state configuration
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model_obj
            )

            return best_model_name, best_model_score

        except Exception as e:
            raise CustomException(e, sys)