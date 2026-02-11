# I/O Data

Pandas menyediakan berbagai fungsi untuk membaca dan menulis data dari berbagai format file.

## CSV Files

### Membaca CSV

```python
import pandas as pd

# Baca CSV dasar
df = pd.read_csv('data.csv')

# Dengan parameter
df = pd.read_csv('data.csv',
                  sep=',',           # Separator
                  header=0,          # Baris header (0 = pertama)
                  index_col=0,       # Kolom sebagai index
                  usecols=['A', 'B'], # Kolom yang dibaca
                  dtype={'A': int},   # Tipe data
                  na_values=['NA', '-'],  # Nilai NA custom
                  nrows=100,         # Jumlah baris
                  skiprows=1,        # Skip baris awal
                  encoding='utf-8')  # Encoding
```

### Menulis CSV

```python
import pandas as pd

df = pd.DataFrame({
    'nama': ['Ahmad', 'Budi', 'Citra'],
    'nilai': [85, 90, 88]
})

# Tulis CSV dasar
df.to_csv('output.csv', index=False)

# Dengan parameter
df.to_csv('output.csv',
          sep=',',
          index=False,        # Tanpa index
          header=True,        # Dengan header
          columns=['nama'],   # Kolom tertentu
          encoding='utf-8')
```

## Excel Files

### Membaca Excel

```python
import pandas as pd

# Baca Excel
df = pd.read_excel('data.xlsx')

# Baca sheet tertentu
df = pd.read_excel('data.xlsx', sheet_name='Sheet1')

# Baca multiple sheets
dfs = pd.read_excel('data.xlsx', sheet_name=['Sheet1', 'Sheet2'])
# dfs adalah dictionary

# Baca semua sheets
dfs = pd.read_excel('data.xlsx', sheet_name=None)
```

### Menulis Excel

```python
import pandas as pd

df = pd.DataFrame({
    'nama': ['Ahmad', 'Budi'],
    'nilai': [85, 90]
})

# Tulis ke Excel
df.to_excel('output.xlsx', index=False, sheet_name='Data')

# Multiple sheets
with pd.ExcelWriter('output.xlsx') as writer:
    df.to_excel(writer, sheet_name='Sheet1', index=False)
    df.to_excel(writer, sheet_name='Sheet2', index=False)
```

## JSON Files

### Membaca JSON

```python
import pandas as pd

# Baca JSON
df = pd.read_json('data.json')

# Dari JSON string
json_str = '[{"nama": "Ahmad", "nilai": 85}, {"nama": "Budi", "nilai": 90}]'
df = pd.read_json(json_str)

# Orientasi berbeda
df = pd.read_json('data.json', orient='records')  # List of dicts
df = pd.read_json('data.json', orient='columns')  # Dict of columns
df = pd.read_json('data.json', orient='index')    # Dict of rows
```

### Menulis JSON

```python
import pandas as pd

df = pd.DataFrame({
    'nama': ['Ahmad', 'Budi'],
    'nilai': [85, 90]
})

# Tulis JSON
df.to_json('output.json', orient='records', indent=2)

# Ke string
json_str = df.to_json(orient='records')
print(json_str)
```

## SQL Database

### Membaca dari SQL

```python
import pandas as pd
import sqlite3

# Koneksi ke database
conn = sqlite3.connect('database.db')

# Baca tabel
df = pd.read_sql('SELECT * FROM mahasiswa', conn)

# Query dengan parameter
df = pd.read_sql(
    'SELECT * FROM mahasiswa WHERE jurusan = ?',
    conn,
    params=['TI']
)

# Baca seluruh tabel
df = pd.read_sql_table('mahasiswa', conn)

conn.close()
```

### Menulis ke SQL

```python
import pandas as pd
import sqlite3

df = pd.DataFrame({
    'nama': ['Ahmad', 'Budi'],
    'nilai': [85, 90]
})

conn = sqlite3.connect('database.db')

# Tulis ke tabel
df.to_sql('nilai_mahasiswa', conn, 
          index=False,
          if_exists='replace')  # 'fail', 'replace', 'append'

conn.close()
```

## HTML Tables

### Membaca HTML

```python
import pandas as pd

# Baca tabel dari HTML (return list of DataFrames)
tables = pd.read_html('https://example.com/table.html')
df = tables[0]  # Tabel pertama

# Dari file lokal
tables = pd.read_html('data.html')

# Dengan matching
tables = pd.read_html('page.html', match='Nama')  # Cari tabel dengan kata "Nama"
```

### Menulis HTML

```python
import pandas as pd

df = pd.DataFrame({
    'nama': ['Ahmad', 'Budi'],
    'nilai': [85, 90]
})

# Ke string HTML
html = df.to_html(index=False)
print(html)

# Ke file
df.to_html('output.html', index=False)
```

## Pickle (Python Object)

```python
import pandas as pd

df = pd.DataFrame({
    'nama': ['Ahmad', 'Budi'],
    'nilai': [85, 90]
})

# Simpan ke pickle
df.to_pickle('data.pkl')

# Baca dari pickle
df_loaded = pd.read_pickle('data.pkl')
```

:::{note}
Pickle menyimpan objek Python secara binary. Ini lebih cepat untuk read/write tapi tidak portable antar bahasa pemrograman.
:::

## Parquet (Columnar Format)

```python
import pandas as pd

df = pd.DataFrame({
    'nama': ['Ahmad', 'Budi'],
    'nilai': [85, 90]
})

# Simpan ke parquet (perlu pyarrow atau fastparquet)
df.to_parquet('data.parquet', engine='pyarrow')

# Baca dari parquet
df_loaded = pd.read_parquet('data.parquet')
```

Parquet sangat efisien untuk data besar dan mendukung kompresi.

## Feather Format

```python
import pandas as pd

df = pd.DataFrame({
    'nama': ['Ahmad', 'Budi'],
    'nilai': [85, 90]
})

# Simpan ke feather
df.to_feather('data.feather')

# Baca dari feather
df_loaded = pd.read_feather('data.feather')
```

Feather sangat cepat untuk read/write dan interoperabel dengan R.

## Clipboard

```python
import pandas as pd

# Baca dari clipboard (copy dari Excel)
df = pd.read_clipboard()

# Tulis ke clipboard
df.to_clipboard(index=False)
```

## Membaca Data Besar

### Chunking

```python
import pandas as pd

# Baca file besar dalam chunks
chunks = pd.read_csv('large_file.csv', chunksize=10000)

# Proses setiap chunk
for chunk in chunks:
    # Proses chunk
    process(chunk)

# Atau kombinasikan
result = pd.concat([process(chunk) for chunk in chunks])
```

### Optimasi Memory

```python
import pandas as pd

# Baca dengan tipe data optimal
df = pd.read_csv('data.csv', dtype={
    'id': 'int32',      # Lebih kecil dari int64
    'kategori': 'category',  # Efisien untuk nilai berulang
    'aktif': 'bool'
})

# Konversi setelah baca
df['kategori'] = df['kategori'].astype('category')
```

## Contoh Praktis

### Pipeline I/O

```python
import pandas as pd

def load_and_process(filepath):
    """Load data, process, dan return DataFrame bersih."""
    # Baca file
    df = pd.read_csv(filepath, encoding='utf-8')
    
    # Bersihkan nama kolom
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    
    # Handle missing values
    df = df.dropna(subset=['id'])
    
    # Konversi tipe
    df['tanggal'] = pd.to_datetime(df['tanggal'])
    
    return df

def save_results(df, output_path):
    """Save DataFrame ke berbagai format."""
    # CSV untuk sharing
    df.to_csv(f'{output_path}.csv', index=False)
    
    # Parquet untuk storage efisien
    df.to_parquet(f'{output_path}.parquet')
    
    # Excel untuk non-technical users
    df.to_excel(f'{output_path}.xlsx', index=False)

# Penggunaan
df = load_and_process('raw_data.csv')
df_processed = df.groupby('kategori').sum()
save_results(df_processed, 'hasil_analisis')
```

### Membaca dari URL

```python
import pandas as pd

# CSV dari URL
url = 'https://example.com/data.csv'
df = pd.read_csv(url)

# JSON dari API
api_url = 'https://api.example.com/data'
df = pd.read_json(api_url)
```

## Latihan

1. Baca file CSV dengan encoding berbeda dan handling missing values
2. Export DataFrame ke multiple sheets dalam satu file Excel
3. Baca data dari SQLite database dan join dengan CSV
4. Implementasikan fungsi yang membaca format otomatis berdasarkan ekstensi file
