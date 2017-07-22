# Results

## Russian corpus

**Baseline:**

Preprocessing: CountVectorizer(ngram_range=(1,5), min_df=4, max_features=100000)
Model: LogisticRegression()
Cross-validation: KFold(shuffle=True, random_state=1)
Results: f1-score = 0.954452393179