Week 6: Built an AI cost predictor web app from scratch.

When businesses build with AI, their biggest headache is budgeting. Huge models are smart but expensive; small ones are cheap but less capable. I wanted to see if we could predict a model’s market price based purely on its "specs"—like size, speed, and intelligence.

Using a Kaggle dataset, I built a machine learning pipeline. Because market prices are wildly inconsistent, I applied a log math transformation to keep data stable. I trained a Random Forest algorithm and refined it with 5-Fold Cross-Validation so the predictions remain reliable, not just lucky guesses.

Finally, I wrapped everything into a high-performance FastAPI web app. Users simply update the model's intelligence scores, size parameters, and response speeds, and the app instantly calculates a fair-market price prediction.

Repository in the comments.

#MachineLearning #DataScience #MLOps #FastAPI