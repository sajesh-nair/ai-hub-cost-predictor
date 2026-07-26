import os
import sys
import uvicorn
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# Initialize FastAPI application
app = FastAPI(title="AI Inference Cost Predictor")

# Configure Jinja2 templates folder mapping
templates = Jinja2Templates(directory="templates")

# 1. Root Route: Serve the input form web UI
@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    # --- FIXED: Using explicit named parameters ---
    return templates.TemplateResponse(
        request=request,
        name="home.html", 
        context={"results": None}
    )

# 2. Prediction Route: Capture form data, execute pipeline, and return result
# --- FIXED: Changed route decorator from "/" to "/predict" to match home.html form action ---
@app.post("/predict", response_class=HTMLResponse)
async def predict_datapoint(
    request: Request,
    aa_intelligence_index: float = Form(...),
    aa_coding_index: float = Form(...),
    composite_benchmark: float = Form(...),
    parameter_count: float = Form(...),
    output_tokens_per_second: float = Form(...),
    time_to_first_token_s: float = Form(...),
    provider: str = Form(...),
    is_open_source: str = Form(...)
):
    try:
        # Map incoming form fields to our CustomData structure
        data = CustomData(
            aa_intelligence_index=aa_intelligence_index,
            aa_coding_index=aa_coding_index,
            composite_benchmark=composite_benchmark,
            parameter_count=parameter_count,
            output_tokens_per_second=output_tokens_per_second,
            time_to_first_token_s=time_to_first_token_s,
            provider=provider,
            is_open_source=(is_open_source == "True")
        )
        
        # Structure into a DataFrame payload
        pred_df = data.get_data_as_data_frame()
        print("Incoming Web UI Data Payload:\n", pred_df)

        # Trigger our frozen prediction pipeline assets
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        
        # Round the final estimated cost to 4 decimal places
        final_cost = round(results[0], 4)

        # --- FIXED: Using explicit named parameters ---
        return templates.TemplateResponse(
            request=request,
            name="home.html", 
            context={"results": final_cost}
        )

    except Exception as e:
        return HTMLResponse(content=f"<h3>Error in pipeline execution: {e}</h3>", status_code=500)

if __name__ == "__main__":
    # Retrieve port from environment or fallback to Hugging Face default 7860
    port = int(os.environ.get("PORT", 7860))
    
    # Launch ASGI web server using uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=port)