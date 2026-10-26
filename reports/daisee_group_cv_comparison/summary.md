# DAiSEE participant-disjoint five-fold comparison

One training run per fold. Mean and sample SD describe variation across folds, not independent repetitions or confidence intervals. Training sets overlap across folds.
Existing BiLSTM results use checkpoint-verified fold paths and default settings; their training seed and historical source hashes were not recorded. New runs use seed 42. Current manifest hashes are checked against protocol.json.
Tabular models use time-aggregated features; sequence models use sequences. This compares the existing pipelines with their default hyperparameters, not an equal-budget hyperparameter search.
Do not rank models until every row has five folds. LSTM here includes temporal attention.

| Model | Completed folds | Accuracy | Precision | Recall | F1 | Specificity | Balanced Acc. | ROC-AUC | PR-AUC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 5/5 | 0.619 ± 0.055 | 0.342 ± 0.059 | 0.427 ± 0.214 | 0.343 ± 0.083 | 0.701 ± 0.159 | 0.564 ± 0.028 | 0.586 ± 0.037 | 0.351 ± 0.032 |
| Random Forest | 5/5 | 0.603 ± 0.073 | 0.344 ± 0.033 | 0.548 ± 0.092 | 0.418 ± 0.024 | 0.619 ± 0.119 | 0.584 ± 0.028 | 0.624 ± 0.031 | 0.352 ± 0.032 |
| XGBoost | 5/5 | 0.584 ± 0.009 | 0.327 ± 0.037 | 0.562 ± 0.112 | 0.409 ± 0.029 | 0.596 ± 0.034 | 0.579 ± 0.041 | 0.607 ± 0.044 | 0.358 ± 0.046 |
| TCN | 5/5 | 0.603 ± 0.072 | 0.335 ± 0.039 | 0.531 ± 0.143 | 0.408 ± 0.066 | 0.616 ± 0.135 | 0.574 ± 0.029 | 0.593 ± 0.045 | 0.343 ± 0.036 |
| GRU | 5/5 | 0.529 ± 0.046 | 0.301 ± 0.063 | 0.603 ± 0.147 | 0.396 ± 0.066 | 0.507 ± 0.090 | 0.555 ± 0.049 | 0.572 ± 0.057 | 0.312 ± 0.066 |
| LSTM + Temporal Attention | 5/5 | 0.574 ± 0.044 | 0.299 ± 0.063 | 0.473 ± 0.114 | 0.364 ± 0.074 | 0.605 ± 0.088 | 0.539 ± 0.026 | 0.557 ± 0.040 | 0.302 ± 0.066 |
| CNN + GRU | 5/5 | 0.632 ± 0.085 | 0.324 ± 0.053 | 0.331 ± 0.107 | 0.315 ± 0.044 | 0.738 ± 0.134 | 0.535 ± 0.033 | 0.555 ± 0.052 | 0.301 ± 0.049 |
| Transformer Encoder | 5/5 | 0.580 ± 0.096 | 0.301 ± 0.026 | 0.455 ± 0.172 | 0.350 ± 0.087 | 0.613 ± 0.163 | 0.534 ± 0.013 | 0.558 ± 0.028 | 0.323 ± 0.024 |
| BiLSTM + Temporal Attention | 5/5 | 0.601 ± 0.031 | 0.303 ± 0.081 | 0.426 ± 0.144 | 0.351 ± 0.098 | 0.656 ± 0.079 | 0.541 ± 0.039 | 0.557 ± 0.024 | 0.307 ± 0.071 |
