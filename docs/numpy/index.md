# NumPy

NumPy (Numerical Python) adalah pustaka fundamental untuk komputasi ilmiah dengan Python. NumPy menyediakan objek array N-dimensi yang powerful dan alat-alat untuk aljabar linear, transformasi Fourier, dan kemampuan bilangan random.

## Mengapa NumPy?

- **Array multidimensi** - Struktur data ndarray yang efisien
- **Broadcasting** - Operasi aritmatika pada array berbeda ukuran
- **Operasi vektorisasi** - Jauh lebih cepat dari Python loop biasa
- **Interoperabilitas** - Basis untuk Pandas, SciPy, scikit-learn, dll

## Instalasi

```bash
pip install numpy
```

## Import NumPy

Konvensi standar untuk mengimport NumPy:

```python
import numpy as np
```

## Daftar Materi

```{toctree}
:maxdepth: 1

array_dasar
indexing_slicing
broadcasting
operasi_matriks
random_dan_io
```

## Contoh Cepat

```python
import numpy as np

# Membuat array
arr = np.array([1, 2, 3, 4, 5])
print(arr)  # [1 2 3 4 5]

# Operasi matematika
print(arr * 2)      # [2 4 6 8 10]
print(arr + 10)     # [11 12 13 14 15]
print(np.sqrt(arr)) # [1. 1.414... 1.732... 2. 2.236...]

# Array 2D (matriks)
matriks = np.array([
    [1, 2, 3],
    [4, 5, 6]
])
print(matriks.shape)  # (2, 3)
```

## Perbandingan: Python List vs NumPy Array

```python
import numpy as np
import time

# Python list
py_list = list(range(1000000))

start = time.time()
py_result = [x * 2 for x in py_list]
print(f"Python list: {time.time() - start:.4f} detik")

# NumPy array
np_arr = np.arange(1000000)

start = time.time()
np_result = np_arr * 2
print(f"NumPy array: {time.time() - start:.4f} detik")
```

NumPy biasanya 10-100x lebih cepat untuk operasi numerik!

## Langkah Selanjutnya

Lanjutkan ke [Array Dasar](array_dasar.md) untuk mempelajari cara membuat dan memanipulasi array NumPy.
