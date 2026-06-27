import os
import sys
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging

class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_columns = [
                'aa_intelligence_index', 'aa_coding_index', 'composite_benchmark',
                'parameter_count', 'output_tokens_per_second', 'time_to_first_token_s'
            ]
            categorical_columns = ['provider', 'is_open_source']

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            # --- THE FIX: Added sparse_output=False ---
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
                ]
            )

            logging.info(f"Numerical columns scaling/imputing: {numerical_columns}")
            logging.info(f"Categorical columns encoding: {categorical_columns}")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
        

    def initiate_data_transformation(self, train_path: str, test_path: str):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            
            numerical_columns = [
                'aa_intelligence_index', 'aa_coding_index', 'composite_benchmark',
                'parameter_count', 'output_tokens_per_second', 'time_to_first_token_s'
            ]
            categorical_columns = ['provider', 'is_open_source']
            target_column_name = "blended_cost_usd_per_1m"
            
            keep_columns = numerical_columns + categorical_columns

            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()
            
            input_feature_train_df = train_df[keep_columns]
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df[keep_columns]
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training and testing dataframes")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            target_feature_train_arr = np.log1p(target_feature_train_df.values)
            target_feature_test_arr = np.log1p(target_feature_test_df.values)

            train_arr = np.c_[input_feature_train_arr, target_feature_train_arr]
            test_arr = np.c_[input_feature_test_arr, target_feature_test_arr]

            logging.info("Saving preprocessing object configuration...")
            
            from src.utils import save_object
            os.makedirs(os.path.dirname(self.data_transformation_config.preprocessor_obj_file_path), exist_ok=True)
            
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)