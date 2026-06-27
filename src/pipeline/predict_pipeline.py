import os
import sys
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features: pd.DataFrame):
        try:
            # 1. Define paths to our frozen production assets
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

            logging.info("Loading preprocessor and model artifacts...")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            logging.info("Transforming incoming web features...")
            # 2. Scale and encode the incoming raw data using the training rules
            scaled_data = preprocessor.transform(features)

            logging.info("Executing prediction loop...")
            # 3. Generate the prediction (returns log-transformed values)
            log_prediction = model.predict(scaled_data)

            # 4. CRITICAL: Invert the log-transformation back to normal USD values!
            # Since we used np.log1p(x), we invert it using np.expm1(y)
            real_prediction = np.expm1(log_prediction)

            return real_prediction

        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    """
    This class is responsible for mapping incoming front-end form values
    into a structured pandas DataFrame that our pipeline understands.
    """
    def __init__(self,
                 aa_intelligence_index: float,
                 aa_coding_index: float,
                 composite_benchmark: float,
                 parameter_count: float,
                 output_tokens_per_second: float,
                 time_to_first_token_s: float,
                 provider: str,
                 is_open_source: bool):
        
        self.aa_intelligence_index = aa_intelligence_index
        self.aa_coding_index = aa_coding_index
        self.composite_benchmark = composite_benchmark
        self.parameter_count = parameter_count
        self.output_tokens_per_second = output_tokens_per_second
        self.time_to_first_token_s = time_to_first_token_s
        self.provider = provider
        self.is_open_source = is_open_source

    def get_data_as_data_frame(self) -> pd.DataFrame:
        try:
            custom_data_input_dict = {
                "aa_intelligence_index": [self.aa_intelligence_index],
                "aa_coding_index": [self.aa_coding_index],
                "composite_benchmark": [self.composite_benchmark],
                "parameter_count": [self.parameter_count],
                "output_tokens_per_second": [self.output_tokens_per_second],
                "time_to_first_token_s": [self.time_to_first_token_s],
                "provider": [self.provider],
                "is_open_source": [self.is_open_source]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)