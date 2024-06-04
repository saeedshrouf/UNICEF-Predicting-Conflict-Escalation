# Define thresholds for models
threshold_transformer = 0.63
threshold_xgboost = 0.71
threshold_ffnn = 0.54

# Initialize classification performance outcome columns
tp['transformer_classifcation_performance_outcome'] = None
tp['xgboost_classifcation_performance_outcome'] = None
tp['ffnn_classifcation_performance_outcome'] = None

# Transformer classification performance outcome
tmp = tp['transformer_classifcation_performance_outcome'].copy()
TP_pos_pred_correct = tp.y_true_transformer & (tp.y_pred_proba_transformer > threshold_transformer)
tmp[TP_pos_pred_correct] = "correctly predicted escalation"
TN_neg_pred_correct = (~tp.y_true_transformer) & (tp.y_pred_proba_transformer <= threshold_transformer)
tmp[TN_neg_pred_correct] = "correctly predicted no escalation"
FP_pos_pred_wrong = (~tp.y_true_transformer) & (tp.y_pred_proba_transformer > threshold_transformer)
tmp[FP_pos_pred_wrong] = "wrongly predicted escalation"
FN_neg_pred_wrong = tp.y_true_transformer & (tp.y_pred_proba_transformer <= threshold_transformer)
tmp[FN_neg_pred_wrong] = "wrongly predicted no escalation"

tp['transformer_classifcation_performance_outcome'] = tmp
tp['transformer_correctness'] = (
    (tp.y_true_transformer & (tp.y_pred_proba_transformer > threshold_transformer)) |
    (~tp.y_true_transformer) & (tp.y_pred_proba_transformer <= threshold_transformer)
).astype(int)

# XGBoost classification performance outcome
tmp = tp['xgboost_classifcation_performance_outcome'].copy()
TP_pos_pred_correct = tp.y_true_xgboost & (tp.y_pred_proba_xgboost > threshold_xgboost)
tmp[TP_pos_pred_correct] = "correctly predicted escalation"
TN_neg_pred_correct = (~tp.y_true_xgboost) & (tp.y_pred_proba_xgboost <= threshold_xgboost)
tmp[TN_neg_pred_correct] = "correctly predicted no escalation"
FP_pos_pred_wrong = (~tp.y_true_xgboost) & (tp.y_pred_proba_xgboost > threshold_xgboost)
tmp[FP_pos_pred_wrong] = "wrongly predicted escalation"
FN_neg_pred_wrong = tp.y_true_xgboost & (tp.y_pred_proba_xgboost <= threshold_xgboost)
tmp[FN_neg_pred_wrong] = "wrongly predicted no escalation"

tp['xgboost_classifcation_performance_outcome'] = tmp
tp['xgboost_correctness'] = (
    (tp.y_true_xgboost & (tp.y_pred_proba_xgboost > threshold_xgboost)) |
    (~tp.y_true_xgboost) & (tp.y_pred_proba_xgboost <= threshold_xgboost)
).astype(int)

# FFNN classification performance outcome
tmp = tp['ffnn_classifcation_performance_outcome'].copy()
TP_pos_pred_correct = tp.y_true_ffnn & (tp.y_pred_proba_ffnn > threshold_ffnn)
tmp[TP_pos_pred_correct] = "correctly predicted escalation"
TN_neg_pred_correct = (~tp.y_true_ffnn) & (tp.y_pred_proba_ffnn <= threshold_ffnn)
tmp[TN_neg_pred_correct] = "correctly predicted no escalation"
FP_pos_pred_wrong = (~tp.y_true_ffnn) & (tp.y_pred_proba_ffnn > threshold_ffnn)
tmp[FP_pos_pred_wrong] = "wrongly predicted escalation"
FN_neg_pred_wrong = tp.y_true_ffnn & (tp.y_pred_proba_ffnn <= threshold_ffnn)
tmp[FN_neg_pred_wrong] = "wrongly predicted no escalation"

tp['ffnn_classifcation_performance_outcome'] = tmp
tp['ffnn_correctness'] = (
    (tp.y_true_ffnn & (tp.y_pred_proba_ffnn > threshold_ffnn)) |
    (~tp.y_true_ffnn) & (tp.y_pred_proba_ffnn <= threshold_ffnn)
).astype(int)
