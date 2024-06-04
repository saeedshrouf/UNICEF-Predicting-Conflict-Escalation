pio.renderers.default = 'notebook'
progress_indicator = 'sowc_social-protection-and-equity__gdp-per-capita-current-us-2010-2019-r_bottom-2-value'
go.Figure(data=go.Choropleth(
    locations=cid['iso3'], text=cid['iso3'], z=cid[progress_indicator],
    colorscale='Greens', autocolorscale=False, reversescale=False,
    marker_line_width=0.5, colorbar_tickprefix='', colorbar_title="GDP per capita"))
