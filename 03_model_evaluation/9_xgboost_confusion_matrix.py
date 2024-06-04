threshold = threshold_xgboost
_ = ConfusionMatrixDisplay.from_predictions(tp.y_true_xgboost, tp.y_pred_proba_xgboost > threshold)
