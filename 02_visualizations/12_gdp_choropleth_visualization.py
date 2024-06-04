df = tp.merge(cid, left_on='iso3', right_on='iso3', how='inner')
fig = []
for model, colors in zip(['y_pred_proba_transformer', 'y_pred_proba_ffnn', 'y_pred_proba_xgboost'], ['Reds', 'Blues', 'Greens']):
    fig += [go.Figure(data=go.Choropleth(
        locations=df['iso3'], text=df['iso3'], z=df[model],
        colorscale=colors, autocolorscale=False, reversescale=False,
        marker_line_width=0.5, colorbar_tickprefix='', colorbar_title=model))]
fig[0].show()
fig[1].show()
fig[2].show()
