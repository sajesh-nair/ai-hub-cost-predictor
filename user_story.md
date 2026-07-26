Week 6: Built an Inference cost predictor web app from scratch.

Budgeting for LLMs is a massive headache for businesses. To solve this, I built a machine learning pipeline that predicts an API's market price based entirely on its technical specs.

Using a Kaggle dataset, I stabilized highly skewed market pricing data with a log transformation, trained a Random Forest Regressor, and utilized 5-Fold Cross-Validation to ensure robust, reliable generalizations. I then deployed the architecture via FastAPI.

What to input to test the live app:

Intelligence/Coding Indices: Standard benchmark reasoning scores (0-100).

Parameter Count: Model size in billions (e.g., 7B or 70B).

Speed & Latency: Tokens Per Second (TPS) and Time to First Token (TTFT).

Live app link and repository in the comments!

#MachineLearning #DataScience #MLOps #FastAPI


Try the live predictor app here:
https://huggingface.co/spaces/sajesh-nair-ai/inference-cost-predictor