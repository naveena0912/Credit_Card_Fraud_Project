

def train_val_test_split(features, labels, data, cfg):
    cut = int(cfg["training"]["train_size"] * len(data))
    train_df, test_df = data.iloc[:cut], data.iloc[cut:]

    # Inner validation from train (time-ordered)
    cut_in = int(cfg["training"]["train_size"] * len(train_df))
    train__df, val_df = train_df.iloc[:cut_in], train_df.iloc[cut_in:]

    X_train , y_train = train__df[features], train__df[labels].astype(int)
    X_val   , y_val   = val_df[features]  , val_df[labels].astype(int)
    X_test  , y_test  = test_df[features] , test_df[labels].astype(int)
    return X_train, X_val, X_test, y_train, y_val, y_test