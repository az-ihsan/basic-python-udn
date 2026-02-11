# Series dan DataFrame

Series dan DataFrame adalah dua struktur data utama dalam Pandas. Mari pelajari cara membuat dan menggunakannya.

## Series

Series adalah array 1D dengan label (index).

### Membuat Series

```python
import pandas as pd
import numpy as np

# Dari list
s1 = pd.Series([10, 20, 30, 40])
print(s1)
# 0    10
# 1    20
# 2    30
# 3    40
# dtype: int64

# Dengan index custom
s2 = pd.Series([10, 20, 30], index=['a', 'b', 'c'])
print(s2)
# a    10
# b    20
# c    30
# dtype: int64

# Dari dictionary
s3 = pd.Series({'x': 100, 'y': 200, 'z': 300})
print(s3)

# Dari NumPy array
s4 = pd.Series(np.arange(5))
print(s4)
```

### Akses Elemen Series

```python
import pandas as pd

s = pd.Series([10, 20, 30, 40], index=['a', 'b', 'c', 'd'])

# Akses dengan label
print(s['b'])    # 20
print(s[['a', 'c']])  # Multiple labels

# Akses dengan posisi
print(s.iloc[1])  # 20
print(s.iloc[0:2])  # Slicing

# Akses dengan kondisi
print(s[s > 20])  # c    30, d    40
```

### Atribut Series

```python
import pandas as pd

s = pd.Series([10, 20, 30], index=['a', 'b', 'c'], name='nilai')

print(s.values)  # [10 20 30]
print(s.index)   # Index(['a', 'b', 'c'], dtype='object')
print(s.dtype)   # int64
print(s.name)    # nilai
print(len(s))    # 3
```

## DataFrame

DataFrame adalah tabel 2D dengan label baris dan kolom.

### Membuat DataFrame

```python
import pandas as pd

# Dari dictionary of lists
data = {
    'nama': ['Ahmad', 'Budi', 'Citra', 'Dani'],
    'umur': [20, 22, 21, 23],
    'jurusan': ['TI', 'SI', 'TI', 'SI'],
    'ipk': [3.5, 3.8, 3.6, 3.9]
}
df = pd.DataFrame(data)
print(df)
#     nama  umur jurusan  ipk
# 0  Ahmad    20      TI  3.5
# 1   Budi    22      SI  3.8
# 2  Citra    21      TI  3.6
# 3   Dani    23      SI  3.9

# Dengan index custom
df = pd.DataFrame(data, index=['m1', 'm2', 'm3', 'm4'])
print(df)
```

### Dari List of Dictionaries

```python
import pandas as pd

data = [
    {'nama': 'Ahmad', 'nilai': 85},
    {'nama': 'Budi', 'nilai': 90},
    {'nama': 'Citra', 'nilai': 88}
]
df = pd.DataFrame(data)
print(df)
```

### Dari NumPy Array

```python
import pandas as pd
import numpy as np

arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
df = pd.DataFrame(arr, 
                  columns=['A', 'B', 'C'],
                  index=['r1', 'r2', 'r3'])
print(df)
#     A  B  C
# r1  1  2  3
# r2  4  5  6
# r3  7  8  9
```

## Atribut DataFrame

```python
import pandas as pd

df = pd.DataFrame({
    'nama': ['Ahmad', 'Budi', 'Citra'],
    'umur': [20, 22, 21],
    'ipk': [3.5, 3.8, 3.6]
})

print(df.shape)    # (3, 3)
print(df.columns)  # Index(['nama', 'umur', 'ipk'], dtype='object')
print(df.index)    # RangeIndex(start=0, stop=3, step=1)
print(df.dtypes)
# nama    object
# umur     int64
# ipk    float64
# dtype: object

print(df.values)   # Array 2D
print(len(df))     # 3
```

## Melihat Data

```python
import pandas as pd

df = pd.DataFrame({
    'A': range(1, 101),
    'B': range(101, 201),
    'C': range(201, 301)
})

# Baris pertama
print(df.head())     # 5 baris pertama
print(df.head(10))   # 10 baris pertama

# Baris terakhir
print(df.tail())     # 5 baris terakhir
print(df.tail(3))    # 3 baris terakhir

# Sampel random
print(df.sample(5))  # 5 baris random

# Info dataset
print(df.info())

# Statistik deskriptif
print(df.describe())
```

## Menambah dan Menghapus Kolom

### Menambah Kolom

```python
import pandas as pd

df = pd.DataFrame({
    'nama': ['Ahmad', 'Budi', 'Citra'],
    'nilai_uts': [80, 85, 90],
    'nilai_uas': [85, 90, 88]
})

# Menambah kolom baru
df['rata_rata'] = (df['nilai_uts'] + df['nilai_uas']) / 2
print(df)

# Dengan assign (return copy)
df_new = df.assign(lulus=df['rata_rata'] >= 80)
print(df_new)

# Menambah kolom dengan nilai konstan
df['tahun'] = 2026
```

### Menghapus Kolom

```python
import pandas as pd

df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6],
    'C': [7, 8, 9]
})

# Hapus satu kolom
df_dropped = df.drop('C', axis=1)
print(df_dropped)

# Hapus beberapa kolom
df_dropped = df.drop(['B', 'C'], axis=1)

# In-place
df.drop('C', axis=1, inplace=True)

# Menggunakan del
del df['B']
```

## Menambah dan Menghapus Baris

### Menambah Baris

```python
import pandas as pd

df = pd.DataFrame({
    'nama': ['Ahmad', 'Budi'],
    'umur': [20, 22]
})

# Menambah baris dengan loc
df.loc[2] = ['Citra', 21]
print(df)

# Concatenate DataFrame
new_row = pd.DataFrame({'nama': ['Dani'], 'umur': [23]})
df = pd.concat([df, new_row], ignore_index=True)
print(df)
```

### Menghapus Baris

```python
import pandas as pd

df = pd.DataFrame({
    'A': [1, 2, 3, 4, 5],
    'B': [10, 20, 30, 40, 50]
})

# Hapus berdasarkan index
df_dropped = df.drop([0, 2])  # Hapus baris index 0 dan 2
print(df_dropped)

# Hapus berdasarkan kondisi
df = df[df['A'] > 2]  # Keep hanya A > 2
print(df)
```

## Mengubah Index

```python
import pandas as pd

df = pd.DataFrame({
    'nama': ['Ahmad', 'Budi', 'Citra'],
    'nilai': [85, 90, 88]
})

# Set kolom sebagai index
df_indexed = df.set_index('nama')
print(df_indexed)
#        nilai
# nama        
# Ahmad     85
# Budi      90
# Citra     88

# Reset index
df_reset = df_indexed.reset_index()
print(df_reset)

# Rename index
df.index = ['m1', 'm2', 'm3']
print(df)
```

## Rename Kolom

```python
import pandas as pd

df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6]
})

# Rename dengan dictionary
df_renamed = df.rename(columns={'A': 'kolom_a', 'B': 'kolom_b'})
print(df_renamed)

# Rename semua kolom
df.columns = ['x', 'y']
print(df)

# Dengan fungsi
df.columns = df.columns.str.upper()
print(df)  # Kolom menjadi X, Y
```

## Sorting

```python
import pandas as pd

df = pd.DataFrame({
    'nama': ['Citra', 'Ahmad', 'Budi'],
    'umur': [21, 20, 22],
    'nilai': [88, 85, 90]
})

# Sort berdasarkan kolom
df_sorted = df.sort_values('nama')
print(df_sorted)

# Sort descending
df_sorted = df.sort_values('nilai', ascending=False)
print(df_sorted)

# Sort multiple kolom
df_sorted = df.sort_values(['umur', 'nilai'], ascending=[True, False])

# Sort berdasarkan index
df_sorted = df.sort_index()
```

## Latihan

1. Buat DataFrame dari data mahasiswa (nama, nim, jurusan, semester, ipk)
2. Tambahkan kolom 'status' berdasarkan IPK (lulus jika IPK >= 3.0)
3. Sortir data berdasarkan jurusan dan IPK
4. Hapus baris dengan IPK < 2.5
