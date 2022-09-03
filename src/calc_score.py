from src.pachage_list import *

def MCRMSE(y_trues, y_preds):
    scores = []
    idxes = y_trues.shape[1]
    for i in range(idxes):
        y_true = y_trues[:,i]
        y_pred = y_preds[:,i]
        score = mean_squared_error(y_true, y_pred, squared=False) # RMSE
        scores.append(score)
    mcrmse_score = np.mean(scores)
    return mcrmse_score, scores


def get_score(y_trues, y_preds):
    mcrmse_score, scores = MCRMSE(y_trues, y_preds)
    return mcrmse_score, scores

def get_result(oof_df, target_cols=['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']):
    labels = oof_df[target_cols].values
    preds = oof_df[[f"pred_{c}" for c in target_cols]].values
    score, scores = get_score(labels, preds)
    return score, scores