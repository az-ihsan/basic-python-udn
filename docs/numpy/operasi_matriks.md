# Operasi Matriks

NumPy menyediakan berbagai fungsi untuk operasi matematika dan aljabar linear pada array dan matriks.

## Operasi Aritmatika Dasar

### Element-wise Operations

```python
import numpy as np

a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7, 8])

print(a + b)   # [ 6  8 10 12]
print(a - b)   # [-4 -4 -4 -4]
print(a * b)   # [ 5 12 21 32]
print(a / b)   # [0.2  0.333...  0.428...  0.5]
print(a ** 2)  # [ 1  4  9 16]
print(np.sqrt(a))  # [1.  1.414...  1.732...  2.]
```

### Operasi pada Matriks

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Element-wise
print(A + B)
# [[ 6  8]
#  [10 12]]

print(A * B)  # Element-wise multiplication
# [[ 5 12]
#  [21 32]]
```

## Perkalian Matriks

### dot product

```python
import numpy as np

# Vektor dot product
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print(np.dot(a, b))  # 32 = 1*4 + 2*5 + 3*6

# Matriks dot product (perkalian matriks)
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print(np.dot(A, B))
# [[19 22]
#  [43 50]]

# Atau dengan operator @
print(A @ B)
# [[19 22]
#  [43 50]]
```

### matmul

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print(np.matmul(A, B))
# [[19 22]
#  [43 50]]
```

## Fungsi Statistik

```python
import numpy as np

arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Agregasi
print(np.sum(arr))       # 45
print(np.mean(arr))      # 5.0
print(np.median(arr))    # 5.0
print(np.std(arr))       # 2.581...
print(np.var(arr))       # 6.666...
print(np.min(arr))       # 1
print(np.max(arr))       # 9

# Per axis
print(np.sum(arr, axis=0))   # Jumlah per kolom: [12 15 18]
print(np.sum(arr, axis=1))   # Jumlah per baris: [ 6 15 24]
print(np.mean(arr, axis=0))  # Mean per kolom: [4. 5. 6.]
print(np.mean(arr, axis=1))  # Mean per baris: [2. 5. 8.]
```

### Fungsi Agregasi Lainnya

```python
import numpy as np

arr = np.array([3, 1, 4, 1, 5, 9, 2, 6])

print(np.argmin(arr))  # 1 (indeks nilai minimum)
print(np.argmax(arr))  # 5 (indeks nilai maksimum)
print(np.sort(arr))    # [1 1 2 3 4 5 6 9]
print(np.argsort(arr)) # [1 3 6 0 2 4 7 5] (indeks untuk sorting)
print(np.cumsum(arr))  # [ 3  4  8  9 14 23 25 31] (cumulative sum)
print(np.prod(arr))    # 6480 (product)
```

## Fungsi Universal (ufunc)

```python
import numpy as np

arr = np.array([0, np.pi/6, np.pi/4, np.pi/3, np.pi/2])

# Trigonometri
print(np.sin(arr))
print(np.cos(arr))
print(np.tan(arr))

# Eksponensial dan logaritma
arr2 = np.array([1, 2, 3, 4, 5])
print(np.exp(arr2))     # [2.718... 7.389... 20.085... 54.598... 148.413...]
print(np.log(arr2))     # [0. 0.693... 1.098... 1.386... 1.609...]
print(np.log10(arr2))   # [0. 0.301... 0.477... 0.602... 0.698...]
print(np.log2(arr2))    # [0. 1. 1.584... 2. 2.321...]

# Power dan square root
print(np.power(arr2, 3))  # [1 8 27 64 125]
print(np.sqrt(arr2))      # [1. 1.414... 1.732... 2. 2.236...]
```

## Aljabar Linear

### Transpose

```python
import numpy as np

A = np.array([[1, 2, 3], [4, 5, 6]])
print(A.T)
# [[1 4]
#  [2 5]
#  [3 6]]
```

### Determinan

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
det = np.linalg.det(A)
print(det)  # -2.0
```

### Inverse

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
A_inv = np.linalg.inv(A)
print(A_inv)
# [[-2.   1. ]
#  [ 1.5 -0.5]]

# Verifikasi: A @ A_inv = I
print(A @ A_inv)
# [[1. 0.]
#  [0. 1.]]
```

### Eigenvalues dan Eigenvectors

```python
import numpy as np

A = np.array([[4, 2], [1, 3]])

eigenvalues, eigenvectors = np.linalg.eig(A)
print("Eigenvalues:", eigenvalues)   # [5. 2.]
print("Eigenvectors:")
print(eigenvectors)
# [[ 0.894...  -0.707...]
#  [ 0.447...   0.707...]]
```

### Solve Linear System (Ax = b)

```python
import numpy as np

# Sistem persamaan:
# 2x + y = 5
# x + 3y = 7

A = np.array([[2, 1], [1, 3]])
b = np.array([5, 7])

x = np.linalg.solve(A, b)
print(x)  # [1.6 1.8]

# Verifikasi
print(A @ x)  # [5. 7.]
```

### Norm

```python
import numpy as np

v = np.array([3, 4])

# L2 norm (Euclidean)
print(np.linalg.norm(v))  # 5.0

# L1 norm
print(np.linalg.norm(v, ord=1))  # 7.0

# Frobenius norm (untuk matriks)
A = np.array([[1, 2], [3, 4]])
print(np.linalg.norm(A, 'fro'))  # 5.477...
```

### Singular Value Decomposition (SVD)

```python
import numpy as np

A = np.array([[1, 2], [3, 4], [5, 6]])

U, S, Vt = np.linalg.svd(A)
print("U shape:", U.shape)   # (3, 3)
print("S:", S)               # Singular values
print("Vt shape:", Vt.shape) # (2, 2)

# Rekonstruksi (approximation)
S_diag = np.zeros((3, 2))
S_diag[:2, :2] = np.diag(S)
A_reconstructed = U @ S_diag @ Vt
print(A_reconstructed)
```

## Operasi Matriks Lainnya

### Trace (jumlah diagonal)

```python
import numpy as np

A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(np.trace(A))  # 15 = 1 + 5 + 9
```

### Rank

```python
import numpy as np

A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(np.linalg.matrix_rank(A))  # 2
```

### Diagonal

```python
import numpy as np

A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Ekstrak diagonal
print(np.diag(A))  # [1 5 9]

# Buat matriks diagonal
print(np.diag([1, 2, 3]))
# [[1 0 0]
#  [0 2 0]
#  [0 0 3]]
```

## Contoh Praktis

### Regresi Linear Sederhana

```python
import numpy as np

# Data
x = np.array([1, 2, 3, 4, 5])
y = np.array([2.1, 4.0, 5.9, 8.1, 9.8])

# Least squares: y = mx + c
# Bentuk matriks: A @ [m, c].T = y
A = np.column_stack([x, np.ones(len(x))])
params, residuals, rank, s = np.linalg.lstsq(A, y, rcond=None)

m, c = params
print(f"y = {m:.2f}x + {c:.2f}")  # y = 1.96x + 0.12
```

### Proyeksi Vektor

```python
import numpy as np

# Proyeksi u ke v
u = np.array([3, 4])
v = np.array([1, 0])

proj = (np.dot(u, v) / np.dot(v, v)) * v
print(proj)  # [3. 0.]
```

## Latihan

1. Hitung determinan dan inverse dari matriks 3x3
2. Selesaikan sistem persamaan linear 3 variabel
3. Hitung eigenvalues dari matriks kovarians
4. Implementasikan PCA sederhana menggunakan SVD
