threshold = threshold_ffnn
_ = ConfusionMatrixDisplay.from_predictions(tp.y_true_ffnn, tp.y_pred_proba_ffnn > threshold)
