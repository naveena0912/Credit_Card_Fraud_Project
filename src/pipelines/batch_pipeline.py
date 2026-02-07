from src.utils.logger import get_logger
from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.data.feature_engineering import engineer_features
from sklearn.model_selection import train_test_split
from src.training.train import train_model
from src.training.evaluate import evaluate_model, evaluate_model_with_calibration
from src.utils.helper import train_val_test_split

def run_batch_pipeline(cfg):
    
    logger = get_logger(__name__)
    logger.info("Running batch pipeline with configuration...")

    # step 1: Data Ingestion
    data = load_data(cfg["paths"]["data_raw"])
    logger.info("Data loaded with shape: %s", data.shape)

    # step 2: Data Preprocessing
    processed_data = preprocess_data(data, cfg["paths"]["data_interim"])

    # step 3: Feature Engineering
    features, labels, data = engineer_features(processed_data, cfg["paths"]["data_processed"])

    # step 4: train_test_split
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(features, labels, data, cfg)

    # step 5: Model Training
    model = train_model(X_train, y_train, cfg)

    # step 6: Model Evaluation
    if cfg["training"]["calibrate_model"]:
        evaluate_model_with_calibration(model, X_val, y_val, cfg)
        evaluate_model_with_calibration(model, X_test, y_test, cfg)
    else:
        val_metrics = evaluate_model(model, X_val, y_val, cfg)
        logger.info("Model Valuation - evaluation metrics: %s", val_metrics)

        test_metrics = evaluate_model(model, X_test, y_test, cfg)
        logger.info("Model Testing - evaluation metrics: %s", test_metrics)

    logger.info("Batch pipeline completed successfully.")
