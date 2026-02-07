from src.data.feature_engineering import engineer_features
from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.serving.inference import predict
from src.serving.model_loader import load_model
from src.utils.logger import get_logger


def run_realtime_pipeline(cfg):
    logger = get_logger(__name__)
    logger.info("Running real-time pipeline with configuration..")

    # step 1: Load deployed model
    model = load_model(cfg["paths"]["models"]+"/fraudulent.pkl")

    # step 2: connect to real-time data source
    data = load_data(cfg["paths"]["data_external"])
    logger.info("Data loaded with shape: %s", data.shape)

    # step 3: Data Preprocessing
    processed_data = preprocess_data(data, cfg["paths"]["data_external_process"])

    # step 4: Feature engineering
    features, labels, data = engineer_features(processed_data, cfg["paths"]["data_external_process"])

    # step 5: Predict fraud probability
    scores = predict(model, data[features])
    
    # step 6: Decision making based on predictions
    data['Class'] = (scores > cfg["evaluation"]["threshold"]).astype(int)
    data.to_csv(cfg["paths"]["data_external"], index=False)
    logger.info("Predictions saved to %s", cfg["paths"]["data_external"])

    for score in scores:
        if score > cfg["evaluation"]["threshold"]:
            logger.warning(f"Fraud detected! Score={score}")