# Results

## Russian corpus

**Baseline:**

Preprocessing: CountVectorizer(ngram_range=(1,5), min_df=4, max_features=100000)<br>
Model: LogisticRegression()<br>
Cross-validation: train_test_split(test_size=0.1, random_state=42)<br>
Results: f1-score = 0.9551<br>

## Ukrainian corpus

**Baseline:**

Preprocessing: TfidfVectorizer(ngram_range=(1,5), max_features=1000000, min_df=4)<br>
Model: SVC(kernel = 'linear', C = 1)<br>
Cross-validation: train_test_split(test_size=0.1, random_state=42)<br>
Results: f1-score = 0.9062<br>
