# Evaluasi Model

Evaluasi yang tepat sangat penting untuk memahami performa model machine learning. Scikit-learn menyediakan berbagai metrik dan teknik evaluasi.

## Metrik Klasifikasi

### Accuracy

```python
from sklearn.metrics import accuracy_score

y_true = [0, 1, 1, 0, 1, 0]
y_pred = [0, 1, 0, 0, 1, 1]

accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy: {accuracy:.4f}")  # 0.6667
```

### Precision, Recall, F1-Score

```python
from sklearn.metrics import precision_score, recall_score, f1_score

y_true = [0, 1, 1, 0, 1, 0, 1, 1]
y_pred = [0, 1, 0, 0, 1, 1, 1, 1]

precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
```

### Classification Report

```python
from sklearn.metrics import classification_report
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred, target_names=iris.target_names))
```

Output:
```
              precision    recall  f1-score   support

      setosa       1.00      1.00      1.00        10
  versicolor       1.00      1.00      1.00         9
   virginica       1.00      1.00      1.00        11

    accuracy                           1.00        30
   macro avg       1.00      1.00      1.00        30
weighted avg       1.00      1.00      1.00        30
```

### Confusion Matrix

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Visualisasi
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()
```

### ROC Curve dan AUC

```python
from sklearn.metrics import roc_curve, roc_auc_score, RocCurveDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Binary classification
X, y = make_classification(n_samples=1000, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

# Probabilitas kelas positif
y_proba = model.predict_proba(X_test)[:, 1]

# ROC AUC
auc = roc_auc_score(y_test, y_proba)
print(f"ROC AUC: {auc:.4f}")

# Plot ROC Curve
RocCurveDisplay.from_predictions(y_test, y_proba)
plt.title(f'ROC Curve (AUC = {auc:.4f})')
plt.show()
```

## Metrik Regresi

### Mean Squared Error (MSE) dan RMSE

```python
from sklearn.metrics import mean_squared_error
import numpy as np

y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]

mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)

print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
```

### Mean Absolute Error (MAE)

```python
from sklearn.metrics import mean_absolute_error

y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]

mae = mean_absolute_error(y_true, y_pred)
print(f"MAE: {mae:.4f}")
```

### R² Score

```python
from sklearn.metrics import r2_score

y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]

r2 = r2_score(y_true, y_pred)
print(f"R² Score: {r2:.4f}")
```

## Cross-Validation

### K-Fold Cross-Validation

```python
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

iris = load_iris()
model = LogisticRegression(max_iter=200)

# 5-Fold CV
scores = cross_val_score(model, iris.data, iris.target, cv=5, scoring='accuracy')

print(f"CV Scores: {scores}")
print(f"Mean: {scores.mean():.4f}")
print(f"Std: {scores.std():.4f}")
```

### Stratified K-Fold

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(model, iris.data, iris.target, cv=skf, scoring='accuracy')
print(f"Stratified CV Mean: {scores.mean():.4f}")
```

### Cross-Validation dengan Multiple Metrics

```python
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

iris = load_iris()
model = LogisticRegression(max_iter=200)

scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']

results = cross_validate(
    model, iris.data, iris.target, 
    cv=5, 
    scoring=scoring,
    return_train_score=True
)

for metric in scoring:
    test_scores = results[f'test_{metric}']
    train_scores = results[f'train_{metric}']
    print(f"{metric}:")
    print(f"  Train: {train_scores.mean():.4f} (+/- {train_scores.std():.4f})")
    print(f"  Test:  {test_scores.mean():.4f} (+/- {test_scores.std():.4f})")
```

## Hyperparameter Tuning

### Grid Search

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

iris = load_iris()

model = RandomForestClassifier(random_state=42)

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7, None],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    model, param_grid, 
    cv=5, 
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(iris.data, iris.target)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.4f}")
```

### Random Search

```python
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint, uniform

model = RandomForestClassifier(random_state=42)

param_distributions = {
    'n_estimators': randint(50, 200),
    'max_depth': randint(3, 10),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10)
}

random_search = RandomizedSearchCV(
    model, param_distributions,
    n_iter=20,
    cv=5,
    scoring='accuracy',
    random_state=42,
    n_jobs=-1
)

random_search.fit(iris.data, iris.target)

print(f"Best parameters: {random_search.best_params_}")
print(f"Best score: {random_search.best_score_:.4f}")
```

## Learning Curves

```python
from sklearn.model_selection import learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np

iris = load_iris()
model = LogisticRegression(max_iter=200)

train_sizes, train_scores, test_scores = learning_curve(
    model, iris.data, iris.target,
    cv=5,
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='accuracy'
)

train_mean = train_scores.mean(axis=1)
train_std = train_scores.std(axis=1)
test_mean = test_scores.mean(axis=1)
test_std = test_scores.std(axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, label='Training score')
plt.plot(train_sizes, test_mean, label='Cross-validation score')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1)
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.title('Learning Curves')
plt.legend()
plt.grid(True)
plt.show()
```

## Validation Curves

```python
from sklearn.model_selection import validation_curve
from sklearn.svm import SVC
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np

iris = load_iris()

param_range = np.logspace(-3, 3, 7)

train_scores, test_scores = validation_curve(
    SVC(), iris.data, iris.target,
    param_name='C',
    param_range=param_range,
    cv=5,
    scoring='accuracy'
)

train_mean = train_scores.mean(axis=1)
test_mean = test_scores.mean(axis=1)

plt.figure(figsize=(10, 6))
plt.semilogx(param_range, train_mean, label='Training score')
plt.semilogx(param_range, test_mean, label='Cross-validation score')
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.title('Validation Curve for SVM')
plt.legend()
plt.grid(True)
plt.show()
```

## Ringkasan Metrik

| Task | Metrik | Kapan Digunakan |
|------|--------|-----------------|
| Klasifikasi | Accuracy | Kelas seimbang |
| Klasifikasi | Precision | Menghindari false positive penting |
| Klasifikasi | Recall | Menghindari false negative penting |
| Klasifikasi | F1-Score | Trade-off precision/recall |
| Klasifikasi | ROC-AUC | Perbandingan model, threshold tuning |
| Regresi | MSE/RMSE | Penalize error besar |
| Regresi | MAE | Robust terhadap outlier |
| Regresi | R² | Interpretasi variance explained |

## Latihan

1. Evaluasi model klasifikasi dengan confusion matrix dan classification report
2. Bandingkan beberapa model menggunakan cross-validation
3. Lakukan hyperparameter tuning dengan GridSearchCV
4. Analisis learning curve untuk mendeteksi overfitting/underfitting
