import numpy as np
from src.utils.logger import get_logger    


def engineer_features(data, output_path):
    # Placeholder for feature engineering logic
    logger = get_logger(__name__)
    logger.info("Engineering features...")
    data["hour_mod24"] = ((data["Time"]//3600) % 24).astype(int)
    data["is_night"] = data["hour_mod24"].isin([0,1,2,3,4,5,22,23]).astype(int)
    data["is_business_hours"] = data["hour_mod24"].between(9, 17).astype(int)
    data["log_amount"] = np.log1p(data["Amount"])

    data.to_csv(output_path, index=False)
    logger.info("Engineered features saved to %s", output_path)

    features = [col for col in data.columns if col.startswith("V")] + ["hour_mod24", "is_night", "is_business_hours", "log_amount"]
    labels = "Class"
    return features, labels, data.sort_values(by="Time")