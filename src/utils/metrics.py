from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

import numpy as np

from src.utils.logger import get_logger

def evaluate_metrics(model, X_test, y_test, metrics_list):
    

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    results = {}

    for metric in metrics_list:
        if metric == "precision":
            results["precision"] = precision_score(y_test, y_pred)
        elif metric == "recall":
            results["recall"] = recall_score(y_test, y_pred)
        elif metric == "f1_score":
            results["f1_score"] = f1_score(y_test, y_pred)
        elif metric == "roc_auc":
            results["roc_auc"] = roc_auc_score(y_test, y_pred)
        elif metric == "average_precision_score":
            results["average_precision_score"] = average_precision_score(y_test, y_proba)

    return results

def evaluate_metrics_with_calibration(model, X_test, y_test):
    logger = get_logger(__name__)
    y_proba = model.predict_proba(X_test)[:, 1]
    logger.info("[XGB-Cal|Pipeline] VAL AUPRC: %f", average_precision_score(y_test, y_proba))
    logger.info("[XGB-Cal|Pipeline] TEST AUPRC: %f", average_precision_score(y_test, y_proba))
    logger.info("[XGB-Cal|Pipeline] ECE: %f", expected_calibration_error(y_test.values, y_proba, n_bins=15))



def expected_calibration_error(y_true, y_proba, n_bins=10):
    """
    Calculate the Expected Calibration Error (ECE).

    Args:
        y_true (pd.Series or np.array): True binary labels.
        y_proba (np.array): Predicted probabilities for the positive class.
        n_bins (int): Number of bins to use for calibration.
    Returns:
        float: Expected Calibration Error.
    """

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(y_proba, bin_edges) - 1
    ece = 0.0
    for i in range(n_bins):
        bin_mask = (idx == i)
        if not np.any(bin_mask):
            continue
        conf = np.mean(y_proba[bin_mask])
        acc = np.mean(y_true[bin_mask])
        ece += np.mean(bin_mask) * np.abs(acc - conf)
    return ece