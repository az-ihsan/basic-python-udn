# GroupBy dan Agregasi

GroupBy memungkinkan kita mengelompokkan data dan menerapkan fungsi agregasi pada setiap grup.

## Konsep GroupBy

GroupBy mengikuti pola: **Split → Apply → Combine**

```
Data Original → Split by Group → Apply Function → Combine Results
```

## GroupBy Dasar

```python
import pandas as pd
import numpy as np

speeds = pd.DataFrame(
    [
        ("bird", "Falconiformes", 389.0),
        ("bird", "Psittaciformes", 24.0),
        ("mammal", "Carnivora", 80.2),
        ("mammal", "Primates", np.nan),
        ("mammal", "Carnivora", 58),
    ],
    index=["falcon", "parrot", "lion", "monkey", "leopard"],
    columns=("class", "order", "max_speed"),
)

print(speeds)
#          class           order  max_speed
# falcon    bird   Falconiformes      389.0
# parrot    bird  Psittaciformes       24.0
# lion    mammal       Carnivora       80.2
# monkey  mammal        Primates        NaN
# leopard mammal       Carnivora       58.0
```

### Membuat GroupBy Object

```python
import pandas as pd
import numpy as np

# DataFrame dari contoh sebelumnya
speeds = pd.DataFrame({
    'class': ['bird', 'bird', 'mammal', 'mammal', 'mammal'],
    'order': ['Falconiformes', 'Psittaciformes', 'Carnivora', 'Primates', 'Carnivora'],
    'max_speed': [389.0, 24.0, 80.2, np.nan, 58.0]
}, index=['falcon', 'parrot', 'lion', 'monkey', 'leopard'])

# Groupby satu kolom
grouped = speeds.groupby('class')
print(type(grouped))  # <class 'pandas.core.groupby.DataFrameGroupBy'>

# Lihat grup
print(grouped.groups)
# {'bird': ['falcon', 'parrot'], 'mammal': ['lion', 'monkey', 'leopard']}
```

## Fungsi Agregasi

### Agregasi Tunggal

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'jurusan': ['TI', 'SI', 'TI', 'SI', 'TI', 'SI'],
    'semester': [4, 4, 6, 6, 4, 6],
    'nilai': [85, 90, 78, 88, 92, 80]
})

# Mean per jurusan
print(df.groupby('jurusan')['nilai'].mean())
# jurusan
# SI    86.0
# TI    85.0
# Name: nilai, dtype: float64

# Sum
print(df.groupby('jurusan')['nilai'].sum())

# Count
print(df.groupby('jurusan')['nilai'].count())

# Min, Max
print(df.groupby('jurusan')['nilai'].min())
print(df.groupby('jurusan')['nilai'].max())
```

### Agregasi Multiple Kolom

```python
import pandas as pd

df = pd.DataFrame({
    'jurusan': ['TI', 'SI', 'TI', 'SI', 'TI', 'SI'],
    'nilai_uts': [80, 85, 75, 90, 88, 78],
    'nilai_uas': [85, 90, 80, 88, 92, 82]
})

# Mean semua kolom numerik
print(df.groupby('jurusan').mean())
#          nilai_uts  nilai_uas
# jurusan                      
# SI       84.333333  86.666667
# TI       81.000000  85.666667
```

## Multiple Agregasi dengan agg()

```python
import pandas as pd

df = pd.DataFrame({
    'jurusan': ['TI', 'SI', 'TI', 'SI', 'TI', 'SI'],
    'nilai': [85, 90, 78, 88, 92, 80]
})

# Multiple fungsi agregasi
print(df.groupby('jurusan')['nilai'].agg(['mean', 'min', 'max', 'count']))
#          mean  min  max  count
# jurusan                       
# SI       86.0   80   90      3
# TI       85.0   78   92      3

# Dengan fungsi custom
print(df.groupby('jurusan')['nilai'].agg(['mean', 'std', lambda x: x.max() - x.min()]))
```

### Agregasi Berbeda per Kolom

```python
import pandas as pd

df = pd.DataFrame({
    'jurusan': ['TI', 'SI', 'TI', 'SI'],
    'mahasiswa': ['A', 'B', 'C', 'D'],
    'nilai_uts': [80, 85, 75, 90],
    'nilai_uas': [85, 90, 80, 88]
})

# Fungsi berbeda untuk kolom berbeda
result = df.groupby('jurusan').agg({
    'mahasiswa': 'count',
    'nilai_uts': ['mean', 'max'],
    'nilai_uas': 'mean'
})
print(result)
```

## GroupBy Multiple Kolom

```python
import pandas as pd

df = pd.DataFrame({
    'jurusan': ['TI', 'TI', 'SI', 'SI', 'TI', 'SI'],
    'semester': [4, 6, 4, 6, 4, 6],
    'nilai': [85, 78, 90, 88, 92, 80]
})

# Group by multiple kolom
print(df.groupby(['jurusan', 'semester'])['nilai'].mean())
# jurusan  semester
# SI       4           90.0
#          6           84.0
# TI       4           88.5
#          6           78.0
# Name: nilai, dtype: float64

# Unstack untuk pivot
print(df.groupby(['jurusan', 'semester'])['nilai'].mean().unstack())
# semester     4     6
# jurusan            
# SI        90.0  84.0
# TI        88.5  78.0
```

## Transformasi

Transform mengembalikan data dengan shape yang sama:

```python
import pandas as pd

df = pd.DataFrame({
    'jurusan': ['TI', 'SI', 'TI', 'SI', 'TI', 'SI'],
    'nilai': [85, 90, 78, 88, 92, 80]
})

# Normalisasi per grup
df['nilai_normalized'] = df.groupby('jurusan')['nilai'].transform(
    lambda x: (x - x.mean()) / x.std()
)
print(df)

# Rank per grup
df['rank'] = df.groupby('jurusan')['nilai'].transform('rank', ascending=False)
print(df)

# Percentage dari total grup
df['pct_of_group'] = df.groupby('jurusan')['nilai'].transform(
    lambda x: x / x.sum() * 100
)
print(df)
```

## Filter Grup

```python
import pandas as pd

df = pd.DataFrame({
    'jurusan': ['TI', 'TI', 'TI', 'SI', 'SI', 'TK'],
    'nilai': [85, 78, 92, 90, 88, 80]
})

# Filter grup dengan rata-rata > 85
result = df.groupby('jurusan').filter(lambda x: x['nilai'].mean() > 85)
print(result)

# Filter grup dengan jumlah anggota > 2
result = df.groupby('jurusan').filter(lambda x: len(x) > 2)
print(result)
```

## Iterasi Grup

```python
import pandas as pd

df = pd.DataFrame({
    'jurusan': ['TI', 'SI', 'TI', 'SI'],
    'nama': ['A', 'B', 'C', 'D'],
    'nilai': [85, 90, 78, 88]
})

# Iterasi setiap grup
for name, group in df.groupby('jurusan'):
    print(f"Jurusan: {name}")
    print(group)
    print()
```

## Contoh Praktis

### Analisis Penjualan

```python
import pandas as pd
import numpy as np

np.random.seed(42)
df = pd.DataFrame({
    'tanggal': pd.date_range('2026-01-01', periods=100),
    'produk': np.random.choice(['A', 'B', 'C'], 100),
    'kategori': np.random.choice(['Elektronik', 'Fashion'], 100),
    'penjualan': np.random.randint(100, 1000, 100)
})

# Total penjualan per produk
print(df.groupby('produk')['penjualan'].sum())

# Rata-rata penjualan per kategori dan produk
print(df.groupby(['kategori', 'produk'])['penjualan'].mean())

# Statistik lengkap per kategori
print(df.groupby('kategori')['penjualan'].describe())

# Top 3 hari dengan penjualan tertinggi per produk
top3_per_produk = df.sort_values('penjualan', ascending=False).groupby('produk').head(3)
print(top3_per_produk)
```

### Akses nth Element

```python
import pandas as pd
import numpy as np

speeds = pd.DataFrame({
    'class': ['bird', 'bird', 'mammal', 'mammal', 'mammal'],
    'order': ['Falconiformes', 'Psittaciformes', 'Carnivora', 'Primates', 'Carnivora'],
    'max_speed': [389.0, 24.0, 80.2, np.nan, 58.0]
}, index=['falcon', 'parrot', 'lion', 'monkey', 'leopard'])

# Ambil elemen kedua (index 1) dari setiap grup
print(speeds.groupby('class')[["order", "max_speed"]].nth(1))
#             order  max_speed
# class                       
# bird   Psittaciformes     24.0
# mammal      Primates        NaN
```

## Latihan

1. Hitung rata-rata IPK per jurusan dan semester
2. Temukan mahasiswa dengan nilai tertinggi di setiap jurusan
3. Hitung persentase mahasiswa lulus (IPK >= 3.0) per jurusan
4. Buat ranking mahasiswa dalam setiap jurusan berdasarkan IPK
