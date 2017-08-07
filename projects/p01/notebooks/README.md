# Results

## Russian corpus

**Baseline:**

Preprocessing: CountVectorizer(ngram_range=(1,5), min_df=4, max_features=100000)<br>
Model: LogisticRegression()<br>
Cross-validation: train_test_split(test_size=0.1, random_state=42)<br>
Results: f1-score = 0.9551<br><br>

Data:<br>
0 - 100872<br>
1 - 100872<br>

## Ukrainian corpus

**Baseline:**

Preprocessing: TfidfVectorizer(analyzer='char', ngram_range=(1,7), max_features=1000000)<br>
Model: LogisticRegression(C = 50)<br>
Cross-validation: train_test_split(test_size=0.1, random_state=42)<br>
Results: f1-score = 0.9252<br><br>

Data:<br>
0 - 15182<br>
1 - 9380<br>

## French corpus

**Baseline:**

Preprocessing: CountVectorizer(ngram_range=(1,3), min_df=4, max_features=10000)<br>
Model: LogisticRegression()<br>
Cross-validation: train_test_split(test_size=0.1, random_state=42)<br>
Results: f1-score = 0.9186<br><br>

Data:<br>
0 - 607958<br>
1 - 607353<br>
Source: movie reviews webpage<br>

## Spanish corpus

**Baseline:**

Preprocessing: CountVectorizer(ngram_range=(1,5), min_df=4, max_features=1000000)<br>
Model: LogisticRegression()<br>
Cross-validation: train_test_split(test_size=0.1, random_state=42)<br>
Results: f1-score = 0.8352<br><br>

Data:<br>
0 - 30096<br>
1 - 12439<br>
Source: movie reviews webpages<br>
