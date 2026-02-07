from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from src.pipelines.batch_pipeline import run_batch_pipeline
from src.pipelines.real_time_pipeline import run_realtime_pipeline
import yaml

from src.utils.logger import setup_logger, get_logger


app = FastAPI(title="Credit Card Fraud Detection API", version="1.0.0")

setup_logger("config/logging.yaml")
logger = get_logger(__name__)
logger.info("Fraud Detection API started")

@app.post("/train")
async def train_model_endpoint():
    """
    Endpoint to trigger batch model training pipeline.

    Returns:
        dict: Status message.
    """
    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)

    
    run_batch_pipeline(cfg)

    return {"status": "Batch training pipeline executed successfully."}

@app.post("/predict")
async def predict_fraud(file: UploadFile=File(...)):
    """
    Endpoint to predict fraud on uploaded data file.

    Args:
        file (UploadFile): Uploaded CSV file containing transaction data.

    Returns:
        dict: Prediction results.
    """
    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)

    file_content = await file.read()
    if not file_content:
        return {"error": "No file uploaded."}
    else:
        # store the file in data external path and run real-time pipeline
        file_location = cfg["paths"]["data_external"]
        with open(file_location, "wb") as f:
            f.write(file_content)        
        run_realtime_pipeline(cfg)

    return FileResponse(file_location, media_type='text/csv', filename="predicted_"+file.filename)