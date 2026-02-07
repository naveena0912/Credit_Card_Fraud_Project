import pandas as pd
import os
import kagglehub
from src.utils.logger import get_logger

def load_data(file_path):
    logger = get_logger(__name__)
    if not os.path.exists(file_path):
        csv_path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
        data = pd.read_csv(f"{csv_path}/creditcard.csv", encoding='utf-8')
        data.to_csv(file_path, index=False)
        logger.info("Data downloaded and saved to %s", file_path)
    else:
        data = pd.read_csv(file_path)
        logger.info("Data loaded from %s", file_path)
    return data