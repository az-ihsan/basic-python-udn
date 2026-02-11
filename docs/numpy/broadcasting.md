# Broadcasting

Broadcasting adalah mekanisme powerful di NumPy yang memungkinkan operasi aritmatika pada array dengan bentuk berbeda. NumPy secara otomatis "memperluas" array yang lebih kecil agar cocok dengan yang lebih besar.

## Konsep Dasar

### Operasi dengan Skalar

```python
import numpy as np

arr = np.array([1, 2, 3, 4, 5])

# Skalar di-broadcast ke semua elemen
print(arr + 10)   # [11 12 13 14 15]
print(arr * 2)    # [ 2  4  6  8 10]
print(arr ** 2)   # [ 1  4  9 16 25]
```

Skalar `10` secara otomatis "diperluas" menjadi `[10, 10, 10, 10, 10]`.

### Operasi Array dengan Array

```python
import numpy as np

a = np.array([1, 2, 3])
b = np.array([10, 20, 30])

# Operasi element-wise
print(a + b)  # [11 22 33]
print(a * b)  # [10 40 90]
```

## Aturan Broadcasting

NumPy membandingkan dimensi dari kanan ke kiri. Dua dimensi kompatibel jika:
1. Nilainya sama, atau
2. Salah satunya adalah 1

### Contoh 1: Array 1D + Array 2D

```python
import numpy as np

# Array 2D (3x3)
matriks = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

# Array 1D (3,)
vektor = np.array([10, 20, 30])

# vektor di-broadcast ke setiap baris
hasil = matriks + vektor
print(hasil)
# [[11 22 33]
#  [14 25 36]
#  [17 28 39]]
```

### Contoh 2: Array Kolom + Array Baris

```python
import numpy as np

# Array kolom (3x1)
kolom = np.array([[1], [2], [3]])

# Array baris (1x3) atau (3,)
baris = np.array([10, 20, 30])

# Broadcasting menghasilkan matriks 3x3
hasil = kolom + baris
print(hasil)
# [[11 21 31]
#  [12 22 32]
#  [13 23 33]]
```

### Contoh 3: Membuat Tabel Perkalian

```python
import numpy as np

# Array kolom 1-10
kolom = np.arange(1, 11).reshape(10, 1)

# Array baris 1-10
baris = np.arange(1, 11)

# Tabel perkalian
tabel = kolom * baris
print(tabel)
# [[  1   2   3   4   5   6   7   8   9  10]
#  [  2   4   6   8  10  12  14  16  18  20]
#  ...
#  [ 10  20  30  40  50  60  70  80  90 100]]
```

## Visualisasi Broadcasting

```
a.shape = (3,)      → diinterpretasi sebagai (1, 3)
b.shape = (3, 1)

Langkah broadcasting:
(1, 3)  +  (3, 1)
   ↓         ↓
(3, 3)  +  (3, 3)  = (3, 3)
```

## Menambah Dimensi

### np.newaxis

```python
import numpy as np

arr = np.array([1, 2, 3])
print(arr.shape)  # (3,)

# Tambah dimensi baru
baris = arr[np.newaxis, :]
print(baris.shape)  # (1, 3)

kolom = arr[:, np.newaxis]
print(kolom.shape)  # (3, 1)
print(kolom)
# [[1]
#  [2]
#  [3]]
```

### reshape

```python
import numpy as np

arr = np.array([1, 2, 3])

# Ubah ke kolom
kolom = arr.reshape(-1, 1)
print(kolom.shape)  # (3, 1)

# Ubah ke baris dengan dimensi eksplisit
baris = arr.reshape(1, -1)
print(baris.shape)  # (1, 3)
```

## Contoh Praktis

### Menghitung Jarak Euclidean

```python
import numpy as np

# Titik A dan B
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Jarak = sqrt(sum((a-b)^2))
jarak = np.sqrt(np.sum((a - b) ** 2))
print(jarak)  # 5.196...

# Jarak dari satu titik ke banyak titik
titik = np.array([0, 0, 0])
titik_lain = np.array([
    [1, 0, 0],
    [0, 2, 0],
    [0, 0, 3],
    [1, 1, 1]
])

# Broadcasting: (1, 3) - (4, 3) = (4, 3)
diff = titik - titik_lain
jarak_semua = np.sqrt(np.sum(diff ** 2, axis=1))
print(jarak_semua)  # [1. 2. 3. 1.732...]
```

### Normalisasi per Kolom

```python
import numpy as np

data = np.array([
    [10, 200, 3000],
    [20, 400, 6000],
    [30, 600, 9000]
])

# Hitung mean dan std per kolom
mean = data.mean(axis=0)  # (3,)
std = data.std(axis=0)    # (3,)

# Z-score normalization dengan broadcasting
normalized = (data - mean) / std
print(normalized)
# [[-1.22474487 -1.22474487 -1.22474487]
#  [ 0.          0.          0.        ]
#  [ 1.22474487  1.22474487  1.22474487]]
```

### Outer Product

```python
import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Outer product menggunakan broadcasting
outer = a[:, np.newaxis] * b[np.newaxis, :]
print(outer)
# [[ 4  5  6]
#  [ 8 10 12]
#  [12 15 18]]

# Atau dengan np.outer
print(np.outer(a, b))
```

## Error Broadcasting

Tidak semua kombinasi bentuk bisa di-broadcast:

```python
import numpy as np

a = np.array([
    [1, 2, 3],
    [4, 5, 6]
])  # (2, 3)

b = np.array([1, 2])  # (2,)

# Ini akan error!
# print(a + b)
# ValueError: operands could not be broadcast together with shapes (2,3) (2,)
```

Solusi: reshape agar kompatibel

```python
import numpy as np

a = np.array([
    [1, 2, 3],
    [4, 5, 6]
])  # (2, 3)

b = np.array([1, 2])  # (2,)

# Ubah b menjadi kolom (2, 1)
b_kolom = b[:, np.newaxis]
print(a + b_kolom)
# [[2 3 4]
#  [6 7 8]]
```

## Tips Broadcasting

1. **Selalu cek shape** sebelum operasi: `print(arr.shape)`
2. **Gunakan np.newaxis** untuk menambah dimensi sesuai kebutuhan
3. **Pikirkan dari kanan ke kiri** saat membandingkan dimensi
4. **Hindari loop** - gunakan broadcasting untuk performa

## Latihan

1. Buat matriks 5x5 di mana setiap elemen adalah jumlah indeks baris dan kolom
2. Hitung jarak setiap titik dalam array ke titik origin (0, 0)
3. Kurangi mean dari setiap baris dalam matriks
4. Buat matriks Vandermonde dari array [1, 2, 3, 4, 5]
