import os
import sys
import pandas as pd
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.pipeline.predict_pipeline import PredictPipeline, CustomData

if __name__ == "__main__":
    try:
        # 1. Trigger the Training Infrastructure
        raw_data_source = os.path.join("notebook", "llm_price_performance_tracker_2026-03-31.csv")
        ingestion = DataIngestion()
        train_path, test_path = ingestion.initiate_data_ingestion(raw_data_source)
        
        print(f"✅ Ingestion Successful!")
        print("-" * 50)
        
        data_transformation = DataTransformation()
        train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(train_path, test_path)
        
        print(f"✅ Data Transformation Successful!")
        print("-" * 50)
        
        model_trainer = ModelTrainer()
        best_model_name, best_model_score = model_trainer.initiate_model_trainer(train_arr, test_arr)
        
        print(f"🚀 Model Training Successful! Winner: {best_model_name} (R2: {best_model_score:.4f})")
        print("=" * 60)
        
        # 2. Simulate a Real-Time User API Call
        print("🔮 Simulating incoming custom data configuration prediction...")
        
        # Creating a custom payload for a hypothetical top-tier proprietary model
        mock_user_input = CustomData(
            aa_intelligence_index=82.5,
            aa_coding_index=79.4,
            composite_benchmark=75.2,
            parameter_count=140.0,       # 140B parameters
            output_tokens_per_second=45.0,
            time_to_first_token_s=0.25,
            provider="OpenAI",
            is_open_source=False         # Commercial closed-source API
        )
        
        # Convert custom input parameters into structured DataFrame
        raw_features_df = mock_user_input.get_data_as_data_frame()
        
        # Fire up the prediction pipeline engine
        predict_pipeline = PredictPipeline()
        predicted_cost = predict_pipeline.predict(raw_features_df)
        
        print("\n📊 --- LIVE INFRASTRUCTURE COST PREDICTION RESULT ---")
        print(f"💡 Estimated Cost: ${predicted_cost[0]:.4f} per 1 Million Tokens")
        print("-----------------------------------------------------")
        
    except Exception as e:
        print(f"❌ Full Pipeline Integration failed: {e}")