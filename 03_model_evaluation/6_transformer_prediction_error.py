tp['transformer_probability_prediction_error'] = np.abs(tp['y_true_transformer'].astype(float) - tp['y_pred_proba_transformer'])
tp[['y_true_transformer', 'y_pred_proba_transformer', 'transformer_probability_prediction_error']]
