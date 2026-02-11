# Model Dasar

Scikit-learn menyediakan berbagai algoritma machine learning untuk klasifikasi, regresi, dan clustering. Halaman ini membahas model-model dasar yang sering digunakan.

## Klasifikasi

### Logistic Regression

Model linear untuk klasifikasi binary dan multiclass:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# Train
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# Probabilitas
y_proba = model.predict_proba(X_test)
print(f"Probabilities (first 3):\n{y_proba[:3]}")
```

### Decision Tree

Model berbasis pohon keputusan:

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)

print(f"Accuracy: {model.score(X_test, y_test):.4f}")

# Feature importance
for name, importance in zip(iris.feature_names, model.feature_importances_):
    print(f"{name}: {importance:.4f}")
```

### Random Forest

Ensemble dari banyak decision trees:

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42
)
model.fit(X_train, y_train)

print(f"Accuracy: {model.score(X_test, y_test):.4f}")

# Feature importance
for name, importance in zip(iris.feature_names, model.feature_importances_):
    print(f"{name}: {importance:.4f}")
```

### K-Nearest Neighbors (KNN)

Klasifikasi berdasarkan tetangga terdekat:

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# KNN sensitif terhadap scale, jadi scale dulu
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train_scaled, y_train)

print(f"Accuracy: {model.score(X_test_scaled, y_test):.4f}")
```

### Support Vector Machine (SVM)

```python
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# SVM juga sensitif terhadap scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = SVC(kernel='rbf', C=1.0, gamma='scale')
model.fit(X_train_scaled, y_train)

print(f"Accuracy: {model.score(X_test_scaled, y_test):.4f}")
```

### Naive Bayes

```python
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

model = GaussianNB()
model.fit(X_train, y_train)

print(f"Accuracy: {model.score(X_test, y_test):.4f}")
```

## Regresi

### Linear Regression

```python
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Generate data
X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")
print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
```

### Ridge Regression (L2 Regularization)

```python
from sklearn.linear_model import Ridge

model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

print(f"R² Score: {model.score(X_test, y_test):.4f}")
```

### Lasso Regression (L1 Regularization)

```python
from sklearn.linear_model import Lasso

model = Lasso(alpha=0.1)
model.fit(X_train, y_train)

print(f"R² Score: {model.score(X_test, y_test):.4f}")
print(f"Non-zero coefficients: {(model.coef_ != 0).sum()}")
```

### Random Forest Regressor

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

X, y = make_regression(n_samples=100, n_features=5, noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print(f"R² Score: {model.score(X_test, y_test):.4f}")
```

## Clustering

### K-Means

```python
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Generate data
X, y_true = make_blobs(n_samples=300, centers=4, random_state=42)

# Clustering
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
y_pred = kmeans.fit_predict(X)

print(f"Cluster centers:\n{kmeans.cluster_centers_}")
print(f"Inertia: {kmeans.inertia_:.2f}")

# Visualisasi
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            c='red', marker='x', s=200, linewidths=3)
plt.title('K-Means Clustering')
plt.show()
```

### Elbow Method untuk Menentukan K

```python
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=300, centers=4, random_state=42)

inertias = []
K_range = range(1, 10)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

# Plot elbow curve
import matplotlib.pyplot as plt
plt.plot(K_range, inertias, 'bx-')
plt.xlabel('k')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()
```

### DBSCAN

Clustering berbasis density:

```python
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons

# Generate data
X, _ = make_moons(n_samples=200, noise=0.05, random_state=42)

# DBSCAN
dbscan = DBSCAN(eps=0.2, min_samples=5)
y_pred = dbscan.fit_predict(X)

print(f"Number of clusters: {len(set(y_pred)) - (1 if -1 in y_pred else 0)}")
print(f"Number of noise points: {(y_pred == -1).sum()}")
```

## Perbandingan Model

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

iris = load_iris()
X, y = iris.data, iris.target

# Scale untuk model yang memerlukannya
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

models = {
    'Logistic Regression': LogisticRegression(max_iter=200),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'SVM': SVC(random_state=42),
    'Naive Bayes': GaussianNB()
}

print("Model Comparison (5-fold CV):\n")
for name, model in models.items():
    # Gunakan scaled data untuk KNN dan SVM
    data = X_scaled if name in ['KNN', 'SVM'] else X
    scores = cross_val_score(model, data, y, cv=5, scoring='accuracy')
    print(f"{name:20s}: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")
```

## Latihan

1. Bandingkan performa Logistic Regression vs Random Forest pada dataset Wine
2. Implementasikan K-Means clustering pada dataset Iris dan visualisasikan hasilnya
3. Buat model regresi untuk memprediksi harga rumah (gunakan dataset synthetic)
4. Tentukan jumlah cluster optimal menggunakan Elbow Method dan Silhouette Score
