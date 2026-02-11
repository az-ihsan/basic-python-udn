# Alur Kerja Machine Learning

Halaman ini menjelaskan alur kerja lengkap dalam membangun model machine learning dengan scikit-learn.

## Tahapan Machine Learning

1. **Definisi Masalah** - Klasifikasi, regresi, atau clustering?
2. **Pengumpulan Data** - Load dan eksplorasi data
3. **Preprocessing** - Bersihkan dan transformasi data
4. **Split Data** - Bagi menjadi train dan test
5. **Training** - Latih model
6. **Evaluasi** - Ukur performa model
7. **Tuning** - Optimasi hyperparameter
8. **Deployment** - Simpan dan gunakan model

## 1. Load Data

```python
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris

# Dari dataset bawaan
iris = load_iris()
X = iris.data
y = iris.target
print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Konversi ke DataFrame untuk eksplorasi
df = pd.DataFrame(X, columns=iris.feature_names)
df['target'] = y
print(df.head())
```

### Load dari File

```python
import pandas as pd

# Dari CSV
df = pd.read_csv('data.csv')

# Pisahkan features dan target
X = df.drop('target_column', axis=1)
y = df['target_column']
```

## 2. Eksplorasi Data

```python
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Informasi dasar
print(df.info())
print(df.describe())

# Cek missing values
print(df.isnull().sum())

# Distribusi target
print(df['target'].value_counts())
```

## 3. Split Data

```python
from sklearn.model_selection import train_test_split

X = df.drop('target', axis=1)
y = df['target']

# Split 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,
    random_state=42,
    stratify=y  # Jaga proporsi kelas
)

print(f"Train size: {len(X_train)}")
print(f"Test size: {len(X_test)}")
```

### Train-Validation-Test Split

```python
from sklearn.model_selection import train_test_split

# Split menjadi train+val dan test
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Split train menjadi train dan validation
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42  # 0.25 * 0.8 = 0.2
)

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
```

## 4. Training Model

```python
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Pilih model
model = LogisticRegression(max_iter=200)

# Training
model.fit(X_train, y_train)

print("Model trained successfully!")
```

## 5. Prediksi

```python
# Prediksi label
y_pred = model.predict(X_test)
print(f"Predictions: {y_pred[:10]}")

# Prediksi probabilitas (untuk classifier)
y_proba = model.predict_proba(X_test)
print(f"Probabilities shape: {y_proba.shape}")
print(f"Sample probabilities: {y_proba[0]}")
```

## 6. Evaluasi

```python
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
```

## Contoh Lengkap: Pipeline

```python
from sklearn.datasets import fetch_openml
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data Titanic
X, y = fetch_openml(
    "titanic", version=1, as_frame=True, return_X_y=True, parser="pandas"
)

# Definisikan kolom
numeric_features = ["age", "fare"]
categorical_features = ["embarked", "pclass"]

# Pipeline untuk numerik
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Pipeline untuk kategorik
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

# Gabungkan preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# Full pipeline
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(solver="liblinear"))
])

# Split dan train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
```

## 7. Cross-Validation

```python
from sklearn.model_selection import cross_val_score, KFold

# K-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
print(f"CV Scores: {scores}")
print(f"Mean: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
```

## 8. Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

# Definisi parameter grid
param_grid = {
    'classifier__C': [0.1, 1, 10],
    'classifier__solver': ['liblinear', 'lbfgs']
}

# Grid Search
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.4f}")

# Gunakan model terbaik
best_model = grid_search.best_estimator_
```

## 9. Simpan Model

```python
import joblib

# Simpan model
joblib.dump(model, 'model.joblib')

# Load model
loaded_model = joblib.load('model.joblib')

# Gunakan model yang di-load
predictions = loaded_model.predict(X_test)
```

## Ringkasan Alur Kerja

```python
# Template workflow
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import joblib

# 1. Load dan prep data
X, y = load_data()

# 2. Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 3. Buat pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', SomeClassifier())
])

# 4. Cross-validation
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5)

# 5. Train final model
pipeline.fit(X_train, y_train)

# 6. Evaluate
y_pred = pipeline.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# 7. Save
joblib.dump(pipeline, 'model.joblib')
```

## Latihan

1. Implementasikan workflow lengkap untuk dataset Wine
2. Bandingkan performa 3 model berbeda dengan cross-validation
3. Lakukan hyperparameter tuning untuk Random Forest
4. Buat pipeline dengan preprocessing dan model
