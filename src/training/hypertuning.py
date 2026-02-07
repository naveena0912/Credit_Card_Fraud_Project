from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV

def time_series_hyperparameter_tuning(model, n_splits=3):
    cv = TimeSeriesSplit(n_splits)

    model_calibrated = CalibratedClassifierCV(estimator=model, cv=cv, method='isotonic')

    return model_calibrated