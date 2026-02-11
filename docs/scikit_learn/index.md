# Scikit-learn

Scikit-learn adalah pustaka open-source Python yang menyediakan alat sederhana dan efisien untuk analisis data prediktif. Pustaka ini mencakup algoritma untuk klasifikasi, regresi, clustering, dimensionality reduction, dan model selection.

## Mengapa Scikit-learn?

- **Konsisten** - API yang seragam untuk semua algoritma
- **Lengkap** - Banyak algoritma ML klasik tersedia
- **Terintegrasi** - Bekerja baik dengan NumPy dan Pandas
- **Terdokumentasi** - Dokumentasi yang sangat baik dengan contoh

## Instalasi

```bash
pip install scikit-learn
```

## Import Scikit-learn

```python
# Import modul yang dibutuhkan
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
```

## Daftar Materi

```{toctree}
:maxdepth: 1

workflow_ml
preprocessing
model_dasar
evaluasi
```

## Alur Kerja Machine Learning

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Data      │ → │ Preprocessing│ → │   Split     │
│   Loading   │    │   & Feature │    │ Train/Test  │
└─────────────┘    └─────────────┘    └──────┬──────┘
                                              │
        ┌─────────────────────────────────────┘
        │
        ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Model     │ → │   Model     │ → │   Model     │
│   Training  │    │   Predict   │    │   Evaluate  │
└─────────────┘    └─────────────┘    └─────────────┘
```

## Contoh Cepat

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Load data
iris = load_iris()
X, y = iris.data, iris.target

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Train model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# 4. Predict
y_pred = model.predict(X_test)

# 5. Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")  # Accuracy: 1.00
```

## API Konsisten Scikit-learn

Semua estimator di scikit-learn mengikuti pola yang sama:

```python
# Inisialisasi model dengan hyperparameter
model = SomeModel(param1=value1, param2=value2)

# Training
model.fit(X_train, y_train)

# Prediksi
y_pred = model.predict(X_test)

# Probabilitas (untuk classifier)
y_proba = model.predict_proba(X_test)

# Transformasi (untuk transformer)
X_transformed = transformer.transform(X)

# Fit dan transform sekaligus
X_transformed = transformer.fit_transform(X)
```

## Dataset Bawaan

Scikit-learn menyediakan beberapa dataset untuk latihan:

```python
from sklearn import datasets

# Classification
iris = datasets.load_iris()
wine = datasets.load_wine()
digits = datasets.load_digits()

# Regression
boston = datasets.load_diabetes()

# Synthetic data
X, y = datasets.make_classification(n_samples=1000, n_features=20)
X, y = datasets.make_regression(n_samples=1000, n_features=10)
X, y = datasets.make_blobs(n_samples=1000, centers=3)
```

## Langkah Selanjutnya

Lanjutkan ke [Alur Kerja ML](workflow_ml.md) untuk mempelajari langkah-langkah lengkap dalam membangun model machine learning.
