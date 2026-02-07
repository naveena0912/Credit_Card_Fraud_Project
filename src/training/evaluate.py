from src.utils.metrics import evaluate_metrics, evaluate_metrics_with_calibration

def evaluate_model(model, X_test, y_test, config):
    metrics = evaluate_metrics(model, X_test, y_test, config["evaluation"]["metrics"])
    return metrics


def evaluate_model_with_calibration(model, X_test, y_test, config):
    evaluate_metrics_with_calibration(model, X_test, y_test, config["evaluation"]["metrics"])
    