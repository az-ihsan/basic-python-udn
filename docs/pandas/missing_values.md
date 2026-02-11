# Missing Values

Data yang hilang (missing values) adalah hal umum dalam analisis data. Pandas menyediakan berbagai alat untuk mendeteksi dan menangani missing values.

## Representasi Missing Values

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'nama': ['Ahmad', 'Budi', None, 'Dani'],
    'umur': [20, np.nan, 21, 23],
    'nilai': [85, 90, np.nan, np.nan]
})

print(df)
#     nama  umur  nilai
# 0  Ahmad  20.0   85.0
# 1   Budi   NaN   90.0
# 2   None  21.0    NaN
# 3   Dani  23.0    NaN
```

Pandas menggunakan `NaN` (Not a Number) untuk merepresentasikan missing values numerik, dan `None` atau `NaN` untuk objek.

## Mendeteksi Missing Values

### isna() dan isnull()

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'A': [1, np.nan, 3],
    'B': [np.nan, 2, 3],
    'C': [1, 2, np.nan]
})

# Deteksi missing values
print(df.isna())  # atau df.isnull()
#        A      B      C
# 0  False   True  False
# 1   True  False  False
# 2  False  False   True

# Jumlah missing per kolom
print(df.isna().sum())
# A    1
# B    1
# C    1
# dtype: int64

# Persentase missing per kolom
print(df.isna().sum() / len(df) * 100)

# Total missing dalam DataFrame
print(df.isna().sum().sum())  # 3
```

### notna()

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'A': [1, np.nan, 3],
    'B': [np.nan, 2, 3]
})

# Deteksi non-missing values
print(df.notna())
#        A      B
# 0   True  False
# 1  False   True
# 2   True   True

# Filter baris tanpa missing
print(df[df['A'].notna()])
```

### any() dan all()

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'A': [1, np.nan, 3],
    'B': [4, 5, 6],
    'C': [np.nan, np.nan, np.nan]
})

# Kolom yang memiliki missing
print(df.isna().any())
# A     True
# B    False
# C     True

# Baris yang memiliki missing
print(df.isna().any(axis=1))
# 0    True
# 1    True
# 2    True

# Kolom yang semua nilainya missing
print(df.isna().all())
# A    False
# B    False
# C     True
```

## Menghapus Missing Values

### dropna()

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'A': [1, np.nan, 3, 4],
    'B': [np.nan, 2, 3, 4],
    'C': [1, 2, np.nan, 4]
})

# Hapus baris dengan missing apapun
print(df.dropna())
#      A    B    C
# 3  4.0  4.0  4.0

# Hapus baris yang semua nilainya missing
print(df.dropna(how='all'))

# Hapus baris dengan missing di kolom tertentu
print(df.dropna(subset=['A']))
#      A    B    C
# 0  1.0  NaN  1.0
# 2  3.0  3.0  NaN
# 3  4.0  4.0  4.0

# Hapus kolom dengan missing
print(df.dropna(axis=1))

# Threshold: minimal non-NA
print(df.dropna(thresh=2))  # Minimal 2 nilai non-NA
```

## Mengisi Missing Values

### fillna()

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'A': [1, np.nan, 3, np.nan],
    'B': [np.nan, 2, np.nan, 4]
})

# Isi dengan nilai konstan
print(df.fillna(0))
#      A    B
# 0  1.0  0.0
# 1  0.0  2.0
# 2  3.0  0.0
# 3  0.0  4.0

# Isi dengan nilai berbeda per kolom
print(df.fillna({'A': 100, 'B': 200}))

# Isi dengan mean
print(df.fillna(df.mean()))

# Isi dengan median
print(df.fillna(df.median()))

# Isi dengan mode (nilai paling sering)
print(df.fillna(df.mode().iloc[0]))
```

### Forward Fill dan Backward Fill

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'nilai': [1, np.nan, np.nan, 4, np.nan, 6]
})

# Forward fill: isi dengan nilai sebelumnya
print(df.ffill())
#    nilai
# 0    1.0
# 1    1.0
# 2    1.0
# 3    4.0
# 4    4.0
# 5    6.0

# Backward fill: isi dengan nilai setelahnya
print(df.bfill())
#    nilai
# 0    1.0
# 1    4.0
# 2    4.0
# 3    4.0
# 4    6.0
# 5    6.0

# Limit jumlah fill
print(df.ffill(limit=1))
```

### interpolate()

Mengisi missing dengan interpolasi:

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'nilai': [1, np.nan, np.nan, 4, np.nan, 6]
})

# Linear interpolation
print(df.interpolate())
#    nilai
# 0    1.0
# 1    2.0
# 2    3.0
# 3    4.0
# 4    5.0
# 5    6.0

# Time-based interpolation (untuk time series)
df_time = pd.DataFrame({
    'nilai': [1, np.nan, 3]
}, index=pd.date_range('2026-01-01', periods=3))
print(df_time.interpolate(method='time'))
```

## Mengganti Nilai

### replace()

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'A': [1, -999, 3, -999],
    'B': ['a', 'NA', 'c', 'NA']
})

# Ganti -999 dengan NaN
df_replaced = df.replace(-999, np.nan)
print(df_replaced)

# Ganti multiple values
df_replaced = df.replace({-999: np.nan, 'NA': np.nan})
print(df_replaced)

# Ganti dengan regex
df['B'] = df['B'].replace(r'^NA$', np.nan, regex=True)
```

## Contoh Praktis

### Strategi Penanganan Missing Values

```python
import pandas as pd
import numpy as np

# Data dengan berbagai jenis missing
df = pd.DataFrame({
    'nama': ['Ahmad', 'Budi', None, 'Dani', 'Eka'],
    'umur': [20, np.nan, 21, 23, np.nan],
    'nilai': [85, 90, np.nan, 75, 88],
    'jurusan': ['TI', 'SI', 'TI', None, 'TI']
})

print("Data original:")
print(df)

# 1. Lihat overview missing
print("\nMissing per kolom:")
print(df.isna().sum())

# 2. Strategi per kolom:
# - nama: hapus baris (wajib ada)
# - umur: isi dengan median
# - nilai: isi dengan mean jurusan (jika ada)
# - jurusan: isi dengan mode

df_clean = df.copy()

# Hapus baris tanpa nama
df_clean = df_clean.dropna(subset=['nama'])

# Isi umur dengan median
df_clean['umur'] = df_clean['umur'].fillna(df_clean['umur'].median())

# Isi nilai dengan mean (sederhana)
df_clean['nilai'] = df_clean['nilai'].fillna(df_clean['nilai'].mean())

# Isi jurusan dengan mode
df_clean['jurusan'] = df_clean['jurusan'].fillna(df_clean['jurusan'].mode()[0])

print("\nData setelah cleaning:")
print(df_clean)
```

### Missing Values dalam GroupBy

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'jurusan': ['TI', 'TI', 'SI', 'SI', 'TI'],
    'nilai': [85, np.nan, 90, np.nan, 88]
})

# Isi missing dengan mean per jurusan
df['nilai'] = df.groupby('jurusan')['nilai'].transform(
    lambda x: x.fillna(x.mean())
)
print(df)
```

### Analisis Pattern Missing

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'A': [1, np.nan, 3, np.nan, 5],
    'B': [np.nan, 2, 3, np.nan, 5],
    'C': [1, 2, np.nan, np.nan, 5]
})

# Baris dengan pattern missing tertentu
missing_pattern = df.isna()
print("Pattern missing:")
print(missing_pattern)

# Baris dengan A dan B missing bersamaan
both_missing = missing_pattern['A'] & missing_pattern['B']
print(f"\nBaris dengan A dan B missing: {both_missing.sum()}")

# Korelasi missing antar kolom
print("\nKorelasi missing:")
print(missing_pattern.corr())
```

## Tips Best Practices

1. **Selalu eksplorasi dulu** - Pahami pola missing sebelum mengisi/menghapus
2. **Dokumentasikan keputusan** - Catat strategi yang digunakan
3. **Pertimbangkan domain** - Pilih metode yang masuk akal untuk data Anda
4. **Jangan isi dengan mean/median secara membabi buta** - Bisa mengurangi variance
5. **Pertimbangkan missing sebagai informasi** - Kadang missing itu sendiri bermakna

## Latihan

1. Buat fungsi untuk melaporkan statistik missing values dalam DataFrame
2. Implementasikan strategi pengisian yang berbeda untuk kolom numerik dan kategorik
3. Deteksi baris dengan lebih dari 50% nilai missing
4. Buat visualisasi pattern missing values (heatmap)
