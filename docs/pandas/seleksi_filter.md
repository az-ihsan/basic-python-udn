# Seleksi dan Filter

Pandas menyediakan berbagai cara untuk memilih dan memfilter data dalam DataFrame.

## Seleksi Kolom

### Satu Kolom

```python
import pandas as pd

df = pd.DataFrame({
    'nama': ['Ahmad', 'Budi', 'Citra'],
    'umur': [20, 22, 21],
    'nilai': [85, 90, 88]
})

# Menggunakan bracket notation
print(df['nama'])
# 0    Ahmad
# 1     Budi
# 2    Citra
# Name: nama, dtype: object

# Menggunakan dot notation
print(df.nama)  # Sama hasilnya
```

### Beberapa Kolom

```python
import pandas as pd

df = pd.DataFrame({
    'nama': ['Ahmad', 'Budi', 'Citra'],
    'umur': [20, 22, 21],
    'nilai': [85, 90, 88]
})

# Pilih beberapa kolom (return DataFrame)
print(df[['nama', 'nilai']])
#     nama  nilai
# 0  Ahmad     85
# 1   Budi     90
# 2  Citra     88

# Simpan list kolom
cols = ['nama', 'umur']
print(df[cols])
```

## Seleksi Baris dengan loc dan iloc

### loc (Label-based)

```python
import pandas as pd

df = pd.DataFrame({
    'nama': ['Ahmad', 'Budi', 'Citra', 'Dani'],
    'umur': [20, 22, 21, 23],
    'nilai': [85, 90, 88, 92]
}, index=['m1', 'm2', 'm3', 'm4'])

# Satu baris
print(df.loc['m1'])
# nama     Ahmad
# umur        20
# nilai       85
# Name: m1, dtype: object

# Beberapa baris
print(df.loc[['m1', 'm3']])

# Range baris (inclusive)
print(df.loc['m1':'m3'])

# Baris dan kolom spesifik
print(df.loc['m1', 'nama'])  # Ahmad
print(df.loc['m1':'m2', ['nama', 'nilai']])
```

### iloc (Integer-based)

```python
import pandas as pd

df = pd.DataFrame({
    'nama': ['Ahmad', 'Budi', 'Citra', 'Dani'],
    'umur': [20, 22, 21, 23],
    'nilai': [85, 90, 88, 92]
})

# Satu baris (by position)
print(df.iloc[0])

# Beberapa baris
print(df.iloc[[0, 2]])

# Range baris (exclusive)
print(df.iloc[0:2])  # Baris 0 dan 1

# Baris dan kolom by position
print(df.iloc[0, 1])  # Baris 0, kolom 1 = 20
print(df.iloc[0:2, 1:3])  # Slice baris dan kolom

# Baris terakhir
print(df.iloc[-1])
```

## Filter Baris dengan Kondisi

### Kondisi Tunggal

```python
import pandas as pd

df = pd.DataFrame({
    'nama': ['Ahmad', 'Budi', 'Citra', 'Dani'],
    'umur': [20, 22, 21, 23],
    'nilai': [85, 90, 88, 92]
})

# Filter dengan kondisi
print(df[df['nilai'] > 85])
#     nama  umur  nilai
# 1   Budi    22     90
# 2  Citra    21     88
# 3   Dani    23     92

# Filter dengan sama dengan
print(df[df['umur'] == 21])

# Filter dengan string
print(df[df['nama'] == 'Ahmad'])
```

### Kondisi Kombinasi

```python
import pandas as pd

df = pd.DataFrame({
    'nama': ['Ahmad', 'Budi', 'Citra', 'Dani'],
    'umur': [20, 22, 21, 23],
    'jurusan': ['TI', 'SI', 'TI', 'SI'],
    'nilai': [85, 90, 88, 92]
})

# AND: gunakan &
print(df[(df['nilai'] >= 85) & (df['umur'] < 23)])

# OR: gunakan |
print(df[(df['jurusan'] == 'TI') | (df['nilai'] > 90)])

# NOT: gunakan ~
print(df[~(df['jurusan'] == 'TI')])  # Bukan TI
```

:::{warning}
Gunakan tanda kurung untuk setiap kondisi saat mengkombinasikan kondisi dengan `&` atau `|`.
:::

### isin()

Filter berdasarkan nilai dalam list:

```python
import pandas as pd

df = pd.DataFrame({
    'nama': ['Ahmad', 'Budi', 'Citra', 'Dani', 'Eka'],
    'jurusan': ['TI', 'SI', 'TI', 'SI', 'TK']
})

# Filter yang jurusannya TI atau SI
print(df[df['jurusan'].isin(['TI', 'SI'])])

# Filter yang nama-nya dalam list
nama_list = ['Ahmad', 'Citra', 'Eka']
print(df[df['nama'].isin(nama_list)])
```

### query()

Cara lebih readable untuk filter:

```python
import pandas as pd

df = pd.DataFrame({
    'nama': ['Ahmad', 'Budi', 'Citra', 'Dani'],
    'umur': [20, 22, 21, 23],
    'nilai': [85, 90, 88, 92]
})

# Menggunakan query
print(df.query('nilai > 85'))
print(df.query('umur >= 21 and nilai > 85'))
print(df.query('nama == "Ahmad"'))

# Dengan variabel
min_nilai = 88
print(df.query('nilai >= @min_nilai'))
```

## Filter String

### Metode String

```python
import pandas as pd

df = pd.DataFrame({
    'nama': ['Ahmad Yani', 'Budi Santoso', 'Citra Dewi', 'Dani Ahmad'],
    'email': ['ahmad@udn.ac.id', 'budi@gmail.com', 'citra@udn.ac.id', 'dani@yahoo.com']
})

# Mengandung substring
print(df[df['nama'].str.contains('Ahmad')])

# Diawali dengan
print(df[df['nama'].str.startswith('B')])

# Diakhiri dengan
print(df[df['email'].str.endswith('udn.ac.id')])

# Case insensitive
print(df[df['nama'].str.lower().str.contains('ahmad')])

# Regex
print(df[df['email'].str.contains(r'@\w+\.com$', regex=True)])
```

## Memilih Berdasarkan Tipe Data

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'nama': ['Ahmad', 'Budi', 'Citra'],
    'umur': [20, 22, 21],
    'ipk': [3.5, 3.8, 3.6],
    'aktif': [True, True, False]
})

# Pilih kolom numerik saja
print(df.select_dtypes(include=[np.number]))

# Pilih kolom object (string) saja
print(df.select_dtypes(include=['object']))

# Exclude tipe tertentu
print(df.select_dtypes(exclude=[np.number]))
```

## Contoh Praktis

### Analisis Data Mahasiswa

```python
import pandas as pd

# Data mahasiswa
df = pd.DataFrame({
    'nama': ['Ahmad', 'Budi', 'Citra', 'Dani', 'Eka'],
    'jurusan': ['TI', 'SI', 'TI', 'SI', 'TI'],
    'semester': [4, 6, 4, 8, 2],
    'ipk': [3.5, 3.2, 3.8, 3.0, 3.6]
})

# Mahasiswa TI semester 4 dengan IPK > 3.5
hasil = df[(df['jurusan'] == 'TI') & 
           (df['semester'] == 4) & 
           (df['ipk'] > 3.5)]
print(hasil)

# Top 3 IPK tertinggi
top3 = df.nlargest(3, 'ipk')
print(top3)

# Bottom 2 IPK terendah
bottom2 = df.nsmallest(2, 'ipk')
print(bottom2)
```

### Filter dengan Kondisi Kompleks

```python
import pandas as pd

df = pd.DataFrame({
    'produk': ['A', 'B', 'C', 'D', 'E'],
    'kategori': ['Elektronik', 'Fashion', 'Elektronik', 'Makanan', 'Fashion'],
    'harga': [1000, 500, 1500, 200, 800],
    'stok': [10, 25, 5, 100, 15]
})

# Produk elektronik mahal dengan stok rendah
filter1 = (df['kategori'] == 'Elektronik') & (df['harga'] > 1000) & (df['stok'] < 10)
print(df[filter1])

# Produk dengan harga antara 500-1000
print(df[(df['harga'] >= 500) & (df['harga'] <= 1000)])

# Atau dengan between
print(df[df['harga'].between(500, 1000)])
```

## Latihan

1. Dari dataset mahasiswa, filter mahasiswa semester > 4 dengan IPK >= 3.5
2. Filter email yang berasal dari domain udn.ac.id
3. Pilih 5 data dengan nilai tertinggi dari setiap jurusan
4. Filter data yang nama-nya mengandung lebih dari satu kata
