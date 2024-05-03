from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
# Cost-Sensitive Logistic Regression
from sklearn.metrics import make_scorer, recall_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.utils.class_weight import compute_class_weight

counter = Counter(y)
print(counter)

"""Weighted Logistic Regression with Scikit-Learn"""

# AUC
# define model
# Sınıf ağırlıklarını belirleme
weight_for_0 = 0.81
weight_for_1 = 0.19

# Modeli tanımlama
weights = {0: weight_for_0, 1: weight_for_1}
model = LogisticRegression(solver='lbfgs', class_weight=weights, max_iter=1000)
# define evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
# summarize performance
print('Mean ROC AUC: %.3f' % mean(scores))
# Mean ROC AUC: 0.965

# Recall
# Model oluşturma
# Recall için özel scorer'ı tanımlama
scorer = make_scorer(recall_score)
# Çapraz doğrulama ile modeli değerlendirme
cv_results = cross_validate(model, X, y, scoring=scorer, cv=cv, n_jobs=-1)
# Recall skorlarını çıkarma
recall_scores = cv_results['test_score']
# Ortalama Recall hesaplama
mean_recall = recall_scores.mean()
print('Mean Recall:', mean_recall)
# Mean Recall: 0.5049786664646924


# calculate class weighting
weighting = compute_class_weight('balanced', [0,1], y)
print(weighting)

