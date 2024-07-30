import os
import sys
from dataclasses import dataclass

from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            model = XGBRegressor()
            params = {
                'learning_rate': [0.1, 0.01, 0.05, 0.001],
                'n_estimators': [8, 16, 32, 64, 128, 256]
            }

            gs = GridSearchCV(model, params, cv=3)
            gs.fit(X_train, y_train)

            best_model = gs.best_estimator_

            logging.info(f"Best model parameters: {gs.best_params_}")
            best_model.fit(X_train, y_train)

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)

            if r2_square < 0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            return r2_square

        except Exception as e:
            raise CustomException(e, sys)
