from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def get_random_forest(cfg):
    return ImbPipeline(steps=[
        ('classifier', RandomForestClassifier(n_estimators=300,
                                              class_weight="balanced",
                                              n_jobs=-1,
                                              random_state=cfg["random_seed"])),
    ])

def get_xgboost(cfg, scale_pos_weight):
    return ImbPipeline(steps=[
        ('classifier', XGBClassifier(n_estimators=300,
                                     max_depth=4,
                                     learning_rate=0.05,
                                     subsample=0.9,
                                     colsample_bytree=0.9,
                                     reg_lambda=1,
                                     tree_method='hist',
                                     scale_pos_weight=scale_pos_weight,
                                     eval_metric="aucpr",
                                     n_jobs=-1,
                                     random_state=cfg["random_seed"])),
    ])