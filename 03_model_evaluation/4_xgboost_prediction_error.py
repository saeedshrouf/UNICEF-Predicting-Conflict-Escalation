tp['xgboost_probability_prediction_error'] = np.abs(tp['y_true_xgboost'].astype(float) - tp['y_pred_proba_xgboost'])
tp[['y_true_xgboost', 'y_pred_proba_xgboost', 'xgboost_probability_prediction_error']]
