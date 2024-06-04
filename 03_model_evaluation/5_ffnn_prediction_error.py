tp['ffnn_probability_prediction_error'] = np.abs(tp['y_true_ffnn'].astype(float) - tp['y_pred_proba_ffnn'])
tp[['y_true_ffnn', 'y_pred_proba_ffnn', 'ffnn_probability_prediction_error']]
