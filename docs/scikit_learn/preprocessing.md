# Preprocessing

Preprocessing adalah langkah penting dalam machine learning untuk menyiapkan data sebelum training. Scikit-learn menyediakan berbagai transformer untuk preprocessing.

## Scaling/Normalization

### StandardScaler

Mengubah data menjadi mean=0 dan std=1:

```python
from sklearn.preprocessing import StandardScaler
import numpy as np

X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Original:")
print(X)
print("\nScaled:")
print(X_scaled)
print(f"\nMean: {X_scaled.mean(axis=0)}")  # [0, 0]
print(f"Std: {X_scaled.std(axis=0)}")    # [1, 1]
```

### MinMaxScaler

Mengubah data ke range [0, 1]:

```python
from sklearn.preprocessing import MinMaxScaler
import numpy as np

X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

print("Scaled (0-1):")
print(X_scaled)
# [[0.   0.  ]
#  [0.33 0.33]
#  [0.67 0.67]
#  [1.   1.  ]]
```

### RobustScaler

Robust terhadap outlier (menggunakan median dan IQR):

```python
from sklearn.preprocessing import RobustScaler
import numpy as np

X = np.array([[1], [2], [3], [4], [100]])  # 100 adalah outlier

standard = StandardScaler().fit_transform(X)
robust = RobustScaler().fit_transform(X)

print("StandardScaler (terpengaruh outlier):")
print(standard.flatten())

print("\nRobustScaler (lebih robust):")
print(robust.flatten())
```

## Encoding Kategorik

### LabelEncoder

Untuk target variable:

```python
from sklearn.preprocessing import LabelEncoder

labels = ['cat', 'dog', 'cat', 'bird', 'dog']

encoder = LabelEncoder()
encoded = encoder.fit_transform(labels)

print(f"Classes: {encoder.classes_}")  # ['bird' 'cat' 'dog']
print(f"Encoded: {encoded}")           # [1 2 1 0 2]

# Inverse transform
original = encoder.inverse_transform([0, 1, 2])
print(f"Decoded: {original}")          # ['bird' 'cat' 'dog']
```

### OrdinalEncoder

Untuk features dengan urutan:

```python
from sklearn.preprocessing import OrdinalEncoder
import numpy as np

X = np.array([['low'], ['medium'], ['high'], ['medium']])

encoder = OrdinalEncoder(categories=[['low', 'medium', 'high']])
X_encoded = encoder.fit_transform(X)

print(X_encoded)
# [[0.]
#  [1.]
#  [2.]
#  [1.]]
```

### OneHotEncoder

Untuk features kategorik nominal:

```python
from sklearn.preprocessing import OneHotEncoder
import numpy as np

X = np.array([['TI'], ['SI'], ['TI'], ['TK']])

encoder = OneHotEncoder(sparse_output=False)
X_encoded = encoder.fit_transform(X)

print(f"Categories: {encoder.categories_}")
print("Encoded:")
print(X_encoded)
# [[0. 1. 0.]
#  [1. 0. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]]
```

## Handling Missing Values

### SimpleImputer

```python
from sklearn.impute import SimpleImputer
import numpy as np

X = np.array([[1, 2], [np.nan, 3], [7, 6], [np.nan, np.nan]])

# Impute dengan mean
imputer_mean = SimpleImputer(strategy='mean')
X_imputed = imputer_mean.fit_transform(X)
print("Mean imputation:")
print(X_imputed)

# Impute dengan median
imputer_median = SimpleImputer(strategy='median')
X_imputed = imputer_median.fit_transform(X)
print("\nMedian imputation:")
print(X_imputed)

# Impute dengan nilai konstan
imputer_const = SimpleImputer(strategy='constant', fill_value=0)
X_imputed = imputer_const.fit_transform(X)
print("\nConstant imputation:")
print(X_imputed)

# Impute dengan most_frequent
imputer_freq = SimpleImputer(strategy='most_frequent')
X_imputed = imputer_freq.fit_transform(X)
print("\nMost frequent imputation:")
print(X_imputed)
```

### KNNImputer

```python
from sklearn.impute import KNNImputer
import numpy as np

X = np.array([[1, 2], [np.nan, 3], [7, 6], [4, np.nan]])

imputer = KNNImputer(n_neighbors=2)
X_imputed = imputer.fit_transform(X)
print("KNN imputation:")
print(X_imputed)
```

## Feature Transformation

### PolynomialFeatures

Membuat fitur polinomial:

```python
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

X = np.array([[2, 3], [3, 4]])

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

print("Original:", X)
print("Polynomial:")
print(X_poly)
# [x1, x2, x1^2, x1*x2, x2^2]
print(f"Feature names: {poly.get_feature_names_out()}")
```

### PowerTransformer

Membuat distribusi lebih normal:

```python
from sklearn.preprocessing import PowerTransformer
import numpy as np

X = np.array([[1], [2], [3], [4], [100]])

pt = PowerTransformer(method='yeo-johnson')
X_transformed = pt.fit_transform(X)

print("Original:")
print(X.flatten())
print("\nTransformed:")
print(X_transformed.flatten())
```

## Feature Selection

### VarianceThreshold

Hapus fitur dengan variance rendah:

```python
from sklearn.feature_selection import VarianceThreshold
import numpy as np

X = np.array([
    [0, 0, 1],
    [0, 1, 0],
    [1, 0, 0],
    [0, 1, 1],
    [0, 1, 0],
    [0, 1, 1]
])

selector = VarianceThreshold(threshold=0.16)
X_selected = selector.fit_transform(X)

print(f"Original shape: {X.shape}")
print(f"Selected shape: {X_selected.shape}")
print(f"Selected features: {selector.get_support()}")
```

### SelectKBest

Pilih K fitur terbaik:

```python
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.datasets import load_iris

iris = load_iris()
X, y = iris.data, iris.target

selector = SelectKBest(score_func=f_classif, k=2)
X_selected = selector.fit_transform(X, y)

print(f"Original features: {X.shape[1]}")
print(f"Selected features: {X_selected.shape[1]}")
print(f"Scores: {selector.scores_}")
print(f"Selected: {selector.get_support()}")
```

## Column Transformer

Menerapkan transformer berbeda ke kolom berbeda:

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import pandas as pd

# Sample data
df = pd.DataFrame({
    'umur': [20, 25, 30, None],
    'gaji': [3000, 4000, 5000, 6000],
    'jurusan': ['TI', 'SI', 'TI', 'TK'],
    'kota': ['Jakarta', 'Bandung', 'Jakarta', 'Surabaya']
})

# Definisi kolom
numeric_features = ['umur', 'gaji']
categorical_features = ['jurusan', 'kota']

# Transformer untuk numerik
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Transformer untuk kategorik
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Gabungkan
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Fit dan transform
X_processed = preprocessor.fit_transform(df)
print(f"Processed shape: {X_processed.shape}")
print(X_processed)
```

## Pipeline

Menggabungkan preprocessing dan model:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load data
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# Buat pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(max_iter=200))
])

# Train
pipeline.fit(X_train, y_train)

# Predict
accuracy = pipeline.score(X_test, y_test)
print(f"Accuracy: {accuracy:.4f}")
```

## Contoh Lengkap

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
import pandas as pd
import numpy as np

# Simulasi data
np.random.seed(42)
n = 100
df = pd.DataFrame({
    'umur': np.random.normal(25, 5, n),
    'pengalaman': np.random.normal(3, 2, n),
    'jurusan': np.random.choice(['TI', 'SI', 'TK'], n),
    'lulus': np.random.choice([0, 1], n)
})

# Tambahkan missing values
df.loc[np.random.choice(n, 5), 'umur'] = np.nan

X = df.drop('lulus', axis=1)
y = df['lulus']

# Preprocessing
numeric_features = ['umur', 'pengalaman']
categorical_features = ['jurusan']

preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]), numeric_features),
    ('cat', Pipeline([
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ]), categorical_features)
])

# Full pipeline
model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Cross-validation
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"CV Accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
```

## Latihan

1. Implementasikan preprocessing untuk dataset dengan campuran numerik dan kategorik
2. Bandingkan StandardScaler vs MinMaxScaler pada model tertentu
3. Buat pipeline dengan berbagai teknik imputation
4. Implementasikan feature selection dalam pipeline
