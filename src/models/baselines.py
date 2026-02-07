from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def get_logistic_regression(cfg):
    return ImbPipeline(steps=[
        ("scaler", StandardScaler(with_mean=True)),
        ('classifier', LogisticRegression(max_iter=2000,
                                          class_weight="balanced",
                                          random_state=cfg["random_seed"])),
    ])