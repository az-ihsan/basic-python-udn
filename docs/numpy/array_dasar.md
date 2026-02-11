# Array Dasar

Array adalah struktur data utama dalam NumPy. Mari pelajari cara membuat dan memanipulasi array.

## Membuat Array

### Dari Python List

```python
import numpy as np

# Array 1D
arr1d = np.array([1, 2, 3, 4, 5])
print(arr1d)        # [1 2 3 4 5]
print(type(arr1d))  # <class 'numpy.ndarray'>

# Array 2D (matriks)
arr2d = np.array([
    [1, 2, 3],
    [4, 5, 6]
])
print(arr2d)
# [[1 2 3]
#  [4 5 6]]

# Array 3D
arr3d = np.array([
    [[1, 2], [3, 4]],
    [[5, 6], [7, 8]]
])
print(arr3d.shape)  # (2, 2, 2)
```

### Fungsi Pembuat Array

```python
import numpy as np

# Array dengan nilai nol
zeros = np.zeros((3, 4))
print(zeros)
# [[0. 0. 0. 0.]
#  [0. 0. 0. 0.]
#  [0. 0. 0. 0.]]

# Array dengan nilai satu
ones = np.ones((2, 3))
print(ones)
# [[1. 1. 1.]
#  [1. 1. 1.]]

# Array dengan nilai tertentu
full = np.full((2, 2), 7)
print(full)
# [[7 7]
#  [7 7]]

# Array identitas
identity = np.eye(3)
print(identity)
# [[1. 0. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]]

# Array kosong (tidak diinisialisasi)
empty = np.empty((2, 3))  # Nilai random dari memori
```

### arange dan linspace

```python
import numpy as np

# arange: mirip range() tapi untuk array
arr = np.arange(0, 10, 2)  # start, stop, step
print(arr)  # [0 2 4 6 8]

# Pangkat 3 dari 0-9
a = np.arange(10) ** 3
print(a)  # [0, 1, 8, 27, 64, 125, 216, 343, 512, 729]

# linspace: angka terdistribusi merata
lin = np.linspace(0, 1, 5)  # start, stop, num
print(lin)  # [0.   0.25 0.5  0.75 1.  ]

# Berguna untuk plotting
x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x)
```

## Atribut Array

```python
import numpy as np

arr = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

# Bentuk array (dimensi)
print(arr.shape)  # (2, 3)

# Jumlah dimensi
print(arr.ndim)   # 2

# Total elemen
print(arr.size)   # 6

# Tipe data elemen
print(arr.dtype)  # int64

# Ukuran tiap elemen (bytes)
print(arr.itemsize)  # 8

# Total memori (bytes)
print(arr.nbytes)    # 48
```

## Tipe Data (dtype)

```python
import numpy as np

# Integer
arr_int = np.array([1, 2, 3], dtype=np.int32)
print(arr_int.dtype)  # int32

# Float
arr_float = np.array([1, 2, 3], dtype=np.float64)
print(arr_float)  # [1. 2. 3.]

# Boolean
arr_bool = np.array([True, False, True])
print(arr_bool.dtype)  # bool

# Complex
arr_complex = np.array([1+2j, 3+4j])
print(arr_complex.dtype)  # complex128

# Konversi tipe data
arr = np.array([1.5, 2.7, 3.9])
arr_int = arr.astype(np.int32)
print(arr_int)  # [1 2 3]
```

### Tipe Data Umum

| Tipe | Deskripsi |
|------|-----------|
| `np.int32` | Integer 32-bit |
| `np.int64` | Integer 64-bit |
| `np.float32` | Float 32-bit |
| `np.float64` | Float 64-bit (default) |
| `np.bool_` | Boolean |
| `np.complex64` | Complex 64-bit |
| `np.complex128` | Complex 128-bit |

## Mengubah Bentuk Array

### reshape

```python
import numpy as np

arr = np.arange(12)
print(arr)  # [ 0  1  2  3  4  5  6  7  8  9 10 11]

# Ubah menjadi 3x4
reshaped = arr.reshape(3, 4)
print(reshaped)
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]]

# Ubah menjadi 2x2x3
reshaped_3d = arr.reshape(2, 2, 3)
print(reshaped_3d.shape)  # (2, 2, 3)

# -1 untuk dimensi otomatis
auto = arr.reshape(4, -1)  # 4 baris, kolom otomatis
print(auto.shape)  # (4, 3)
```

### flatten dan ravel

```python
import numpy as np

arr = np.array([[1, 2, 3], [4, 5, 6]])

# flatten: copy ke 1D
flat = arr.flatten()
print(flat)  # [1 2 3 4 5 6]

# ravel: view ke 1D (lebih efisien)
rav = arr.ravel()
print(rav)  # [1 2 3 4 5 6]
```

### transpose

```python
import numpy as np

arr = np.array([
    [1, 2, 3],
    [4, 5, 6]
])
print(arr.shape)  # (2, 3)

# Transpose (tukar baris dan kolom)
trans = arr.T
print(trans)
# [[1 4]
#  [2 5]
#  [3 6]]
print(trans.shape)  # (3, 2)
```

## Menggabungkan Array

### concatenate

```python
import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Gabungkan 1D
gabung = np.concatenate([a, b])
print(gabung)  # [1 2 3 4 5 6]

# Gabungkan 2D
arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([[5, 6], [7, 8]])

# Sepanjang axis 0 (baris)
gabung0 = np.concatenate([arr1, arr2], axis=0)
print(gabung0)
# [[1 2]
#  [3 4]
#  [5 6]
#  [7 8]]

# Sepanjang axis 1 (kolom)
gabung1 = np.concatenate([arr1, arr2], axis=1)
print(gabung1)
# [[1 2 5 6]
#  [3 4 7 8]]
```

### vstack dan hstack

```python
import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Vertical stack (tumpuk vertikal)
v = np.vstack([a, b])
print(v)
# [[1 2 3]
#  [4 5 6]]

# Horizontal stack (gabung horizontal)
h = np.hstack([a, b])
print(h)  # [1 2 3 4 5 6]
```

## Memisahkan Array

```python
import numpy as np

arr = np.arange(12).reshape(4, 3)
print(arr)
# [[ 0  1  2]
#  [ 3  4  5]
#  [ 6  7  8]
#  [ 9 10 11]]

# Split menjadi 2 bagian
bagian = np.split(arr, 2, axis=0)
print(bagian[0])
# [[0 1 2]
#  [3 4 5]]
print(bagian[1])
# [[ 6  7  8]
#  [ 9 10 11]]

# Vertical split
vsplit = np.vsplit(arr, 2)

# Horizontal split
arr2 = np.arange(12).reshape(3, 4)
hsplit = np.hsplit(arr2, 2)
```

## Copy vs View

```python
import numpy as np

arr = np.array([1, 2, 3, 4, 5])

# View (berbagi data dengan original)
view = arr.view()
view[0] = 100
print(arr[0])  # 100 - original berubah!

# Copy (independen dari original)
arr = np.array([1, 2, 3, 4, 5])
copy = arr.copy()
copy[0] = 100
print(arr[0])  # 1 - original tidak berubah
```

:::{warning}
Slicing di NumPy membuat **view**, bukan copy. Modifikasi pada slice akan mempengaruhi array original.
:::

```python
import numpy as np

arr = np.array([1, 2, 3, 4, 5])
slice_arr = arr[1:4]  # Ini adalah view
slice_arr[0] = 99
print(arr)  # [ 1 99  3  4  5] - original berubah!
```

## Latihan

1. Buat array 3x3 berisi angka 1-9
2. Buat array 4x4 dengan diagonal utama berisi 1-4
3. Buat array 2x3x4 (24 elemen) dari angka 0-23
4. Gabungkan dua array 2x2 menjadi array 4x2 dan 2x4
