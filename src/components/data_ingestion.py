import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from src.exception import CustomException
from src.logger import logging

class DataIngestionConfig:
    # Defining where our production artifacts will be stored
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "raw.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self, source_csv_path: str):
        logging.info("Starting the data ingestion process...")
        try:
            # 1. Read the live benchmark dataset we explored in EDA
            df = pd.read_csv(source_csv_path)
            logging.info('Successfully read the raw tracker CSV file into a DataFrame')

            # --- THE FIX: Drop rows where the target value is missing ---
            df = df.dropna(subset=["blended_cost_usd_per_1m"])
            logging.info(f"Dropped records with missing targets. Rows remaining for pipeline: {len(df)}")

            # 2. Create the artifacts directory if it doesn't exist
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # 3. Export a backup copy of the filtered raw data to artifacts
            df.to_csv(self.ingestion_config.raw_data_path, index=False)

            # 4. Perform the Train-Test Split (80% train, 20% test)
            logging.info("Initiating train-test split...")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # 5. Save the clean train and test splits as separate CSVs
            train_set.to_csv(self.ingestion_config.train_data_path, index=False)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False)

            logging.info("Data ingestion and splitting successfully completed!")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)