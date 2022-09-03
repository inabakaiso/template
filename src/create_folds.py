from src.pachage_list import *
from sklearn import model_selection

def create_folds(df: pd.DataFrame, num_folds: int, seed: int, target_cols, groups=None, split_type=None):
    if split_type == "kfold":
        kf = model_selection.KFold(n_splits=num_folds)
        for fold, (_, val_index) in enumerate(kf.split(X=df)):
            df.loc[val_index, 'kfold'] = int(fold)
    elif split_type == "stratified_kfold":
        gkf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
        for fold, (_, val_id) in enumerate(gkf.split(X=df, y=df[target_cols])):
            df.loc[val_id, "kfold"] = fold
    elif split_type == "group_kfold":
        gkf = GroupKFold(n_splits=num_folds)
        gskf = StratifiedGroupKFold(n_splits=num_folds)
        for fold, (_, val_index) in enumerate(gskf.split(df[target_cols],df[groups])):
            df.loc[val_index , "kfold"] = int(fold)
    elif split_type == "gs_kfold":
        gskf = StratifiedGroupKFold(n_splits=num_folds)
        for fold, (_, val_index) in enumerate(gskf.split(df[groups])):
            df.loc[val_index , "kfold"] = int(fold)
    elif split_type == "multi_label_skfold":
        mlkf = MultilabelStratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
        for fold, (_, val_index) in enumerate(mlkf.split(df, df[target_cols])):
            df.loc[val_index, 'kfold'] = int(fold)
    return df