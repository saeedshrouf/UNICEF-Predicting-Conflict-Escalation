threshold = threshold_transformer
_ = ConfusionMatrixDisplay.from_predictions(tp.y_true_transformer, tp.y_pred_proba_transformer > threshold)
