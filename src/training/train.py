
from src.models.baselines import get_logistic_regression
from src.models.ensemble import get_random_forest, get_xgboost
from src.serving.model_loader import save_model
from src.training.hypertuning import time_series_hyperparameter_tuning


def train_model(X_train, y_train, cfg):
    """
    Train the model with the provided training data.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        cfg: Configuration parameters.

    Returns:
        Trained model.
    """
    if cfg["model"]["type"] == "logistic_regression": 
        model = get_logistic_regression(cfg["training"])
    elif cfg["model"]["type"] == "random_forest": 
        model = get_random_forest(cfg["training"])
    elif cfg["model"]["type"] == "xgboost":
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        model = get_xgboost(cfg["training"], scale_pos_weight)
    else: 
        raise ValueError("Unsupported model type") 
    
    if cfg["training"]["calibrate_model"]:
        model = time_series_hyperparameter_tuning(model, n_splits=3)
        
    model.fit(X_train, y_train)
    save_model(model, cfg["paths"]["models"]+"/fraudulent.pkl")
    return model