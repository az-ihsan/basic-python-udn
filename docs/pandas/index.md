# Pandas

Pandas adalah pustaka open source yang menyediakan struktur data dan alat analisis data yang powerful untuk Python. Pandas sangat cocok untuk bekerja dengan data tabular (seperti spreadsheet atau database).

## Mengapa Pandas?

- **DataFrame** - Struktur data 2D dengan label baris dan kolom
- **Fleksibel** - Mudah membaca berbagai format file (CSV, Excel, SQL, dll)
- **Powerful** - Operasi groupby, merge, reshape yang efisien
- **Terintegrasi** - Bekerja baik dengan NumPy, matplotlib, scikit-learn

## Instalasi

```bash
pip install pandas
```

## Import Pandas

Konvensi standar untuk mengimport Pandas:

```python
import pandas as pd
import numpy as np
```

## Daftar Materi

```{toctree}
:maxdepth: 1

series_dataframe
seleksi_filter
groupby_agg
merge_join
missing_values
io_data
```

## Struktur Data Utama

### Series

Array 1D dengan label (index):

```python
import pandas as pd

# Membuat Series
s = pd.Series([1, 3, 5, 7, 9])
print(s)
# 0    1
# 1    3
# 2    5
# 3    7
# 4    9
# dtype: int64

# Series dengan index custom
s = pd.Series([1, 3, 5], index=['a', 'b', 'c'])
print(s['b'])  # 3
```

### DataFrame

Tabel 2D dengan label baris dan kolom:

```python
import pandas as pd

# Membuat DataFrame dari dictionary
data = {
    'nama': ['Ahmad', 'Budi', 'Citra'],
    'umur': [20, 22, 21],
    'nilai': [85, 90, 88]
}
df = pd.DataFrame(data)
print(df)
#     nama  umur  nilai
# 0  Ahmad    20     85
# 1   Budi    22     90
# 2  Citra    21     88
```

## Contoh Cepat

```python
import pandas as pd

# Membaca CSV
df = pd.read_csv('data.csv')

# Melihat data awal
print(df.head())

# Informasi dataset
print(df.info())
print(df.describe())

# Seleksi kolom
print(df['nama'])

# Filter baris
print(df[df['nilai'] > 80])

# Groupby dan agregasi
print(df.groupby('jurusan')['nilai'].mean())
```

## Langkah Selanjutnya

Lanjutkan ke [Series dan DataFrame](series_dataframe.md) untuk mempelajari struktur data Pandas secara mendalam.
