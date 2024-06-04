cid['GDP-per-capita'] = cid['sowc_social-protection-and-equity__gdp-per-capita-current-us-2010-2019-r_bottom-2-value'] > cid['sowc_social-protection-and-equity__gdp-per-capita-current-us-2010-2019-r_bottom-2-value'].median()
cid['GDP-per-capita_values'] = cid['sowc_social-protection-and-equity__gdp-per-capita-current-us-2010-2019-r_bottom-2-value']
df = tp.merge(cid, left_on='iso3', right_on='iso3', how='inner')
high_GDP = df[df['GDP-per-capita'] == True]
((df['GDP-per-capita'] == True).sum(), (df['GDP-per-capita'] == False).sum())
