# Indexing dan Slicing

Indexing dan slicing memungkinkan kita mengakses dan memodifikasi elemen array. NumPy mendukung berbagai cara pengindeksan yang powerful.

## Indexing Dasar

### Array 1D

```python
import numpy as np

data = np.array([1, 2, 3])

# Akses elemen tunggal
print(data[1])   # 2

# Indeks negatif (dari belakang)
print(data[-1])  # 3
print(data[-2])  # 2
```

### Array 2D

```python
import numpy as np

arr = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

# Akses elemen: arr[baris, kolom]
print(arr[0, 0])  # 1
print(arr[1, 2])  # 6
print(arr[2, 1])  # 8

# Akses baris lengkap
print(arr[0])     # [1 2 3]
print(arr[1])     # [4 5 6]

# Akses kolom lengkap
print(arr[:, 0])  # [1 4 7]
print(arr[:, 1])  # [2 5 8]
```

## Slicing

### Array 1D

```python
import numpy as np

a = np.arange(10) ** 3
print(a)  # [0, 1, 8, 27, 64, 125, 216, 343, 512, 729]

# Slicing dasar: arr[start:stop]
print(a[2:5])   # [8 27 64]

# Dari awal sampai indeks tertentu
print(a[:5])    # [0 1 8 27 64]

# Dari indeks tertentu sampai akhir
print(a[5:])    # [125 216 343 512 729]

# Dengan step: arr[start:stop:step]
print(a[::2])   # [0 8 64 216 512] - setiap 2 elemen
print(a[1::2])  # [1 27 125 343 729] - mulai dari indeks 1

# Reverse array
print(a[::-1])  # [729 512 343 216 125 64 27 8 1 0]

# Modifikasi dengan slice
a[:6:2] = 1000
print(a)  # [1000, 1, 1000, 27, 1000, 125, 216, 343, 512, 729]
```

### Array 2D

```python
import numpy as np

arr = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12]
])

# Slice baris dan kolom
print(arr[0:2, 1:3])
# [[2 3]
#  [6 7]]

# Semua baris, kolom tertentu
print(arr[:, 1:3])
# [[ 2  3]
#  [ 6  7]
#  [10 11]]

# Baris tertentu, semua kolom
print(arr[1:, :])
# [[ 5  6  7  8]
#  [ 9 10 11 12]]

# Baris ganjil
print(arr[::2, :])
# [[1  2  3  4]
#  [9 10 11 12]]
```

## Fancy Indexing

Menggunakan array integer sebagai indeks:

```python
import numpy as np

arr = np.arange(10, 100, 10)
print(arr)  # [10 20 30 40 50 60 70 80 90]

# Akses beberapa indeks sekaligus
indeks = [0, 2, 5]
print(arr[indeks])  # [10 30 60]

# Dengan array NumPy
idx = np.array([1, 3, 5, 7])
print(arr[idx])  # [20 40 60 80]

# Untuk 2D
arr2d = np.arange(12).reshape(3, 4)
print(arr2d)
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]]

# Akses elemen (0,1), (1,2), (2,3)
baris = [0, 1, 2]
kolom = [1, 2, 3]
print(arr2d[baris, kolom])  # [1 6 11]
```

## Boolean Indexing

Menggunakan kondisi boolean untuk memfilter:

```python
import numpy as np

arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Buat mask boolean
mask = arr > 5
print(mask)  # [False False False False False True True True True True]

# Filter dengan mask
print(arr[mask])  # [6 7 8 9 10]

# Langsung dalam satu ekspresi
print(arr[arr > 5])  # [6 7 8 9 10]
print(arr[arr % 2 == 0])  # [2 4 6 8 10] - bilangan genap

# Kondisi kombinasi
print(arr[(arr > 3) & (arr < 8)])  # [4 5 6 7]
print(arr[(arr < 3) | (arr > 8)])  # [1 2 9 10]
```

### Boolean Indexing pada Array 2D

```python
import numpy as np

arr = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

# Elemen lebih dari 5
print(arr[arr > 5])  # [6 7 8 9]

# Modifikasi dengan kondisi
arr[arr > 5] = 0
print(arr)
# [[1 2 3]
#  [4 5 0]
#  [0 0 0]]
```

## np.where

Kondisi dengan nilai pengganti:

```python
import numpy as np

arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# np.where(kondisi, nilai_jika_true, nilai_jika_false)
hasil = np.where(arr > 5, arr, 0)
print(hasil)  # [0 0 0 0 0 6 7 8 9 10]

# Mengganti nilai genap dengan -1
hasil = np.where(arr % 2 == 0, -1, arr)
print(hasil)  # [1 -1 3 -1 5 -1 7 -1 9 -1]

# Hanya mendapatkan indeks
indeks = np.where(arr > 5)
print(indeks)  # (array([5, 6, 7, 8, 9]),)
print(indeks[0])  # [5 6 7 8 9]
```

## Iterasi pada Array

```python
import numpy as np

a = np.arange(10) ** 3
print(a)  # [0, 1, 8, 27, 64, 125, 216, 343, 512, 729]

# Iterasi elemen
for i in a:
    print(i ** (1/3.))

# Untuk array multidimensi
arr = np.array([[1, 2], [3, 4], [5, 6]])

# Iterasi baris
for baris in arr:
    print(baris)

# Iterasi semua elemen
for elemen in arr.flat:
    print(elemen, end=" ")  # 1 2 3 4 5 6
```

## Contoh Praktis

### Menormalisasi Data

```python
import numpy as np

data = np.array([10, 20, 30, 40, 50])

# Min-max normalization ke [0, 1]
normalized = (data - data.min()) / (data.max() - data.min())
print(normalized)  # [0.   0.25 0.5  0.75 1.  ]
```

### Mengganti Nilai Outlier

```python
import numpy as np

data = np.array([1, 2, 100, 3, 4, 200, 5])

# Ganti nilai > 10 dengan median
median = np.median(data[data <= 10])
data[data > 10] = median
print(data)  # [1 2 3 3 4 3 5]
```

### Memilih Baris dengan Kondisi

```python
import numpy as np

# Data mahasiswa: [nilai_uts, nilai_uas, nilai_tugas]
nilai = np.array([
    [80, 85, 90],
    [60, 65, 70],
    [90, 95, 92],
    [50, 55, 60]
])

# Rata-rata per mahasiswa
rata = nilai.mean(axis=1)
print(rata)  # [85. 65. 92.33... 55.]

# Mahasiswa dengan rata-rata > 70
lulus = nilai[rata > 70]
print(lulus)
# [[80 85 90]
#  [90 95 92]]
```

## Latihan

1. Dari array 1-100, ambil semua bilangan yang habis dibagi 7
2. Buat matriks 5x5, ambil submatriks 3x3 di tengah
3. Dari matriks 4x4, ganti semua nilai diagonal dengan 0
4. Filter matriks untuk mengambil baris yang jumlahnya > 10
