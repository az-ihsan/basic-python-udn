# Merge dan Join

Pandas menyediakan berbagai cara untuk menggabungkan DataFrame, mirip dengan operasi JOIN di SQL.

## Merge

Fungsi `merge()` digunakan untuk menggabungkan DataFrame berdasarkan kolom kunci.

### Inner Merge (Default)

```python
import pandas as pd

# DataFrame mahasiswa
mahasiswa = pd.DataFrame({
    'nim': ['001', '002', '003', '004'],
    'nama': ['Ahmad', 'Budi', 'Citra', 'Dani']
})

# DataFrame nilai
nilai = pd.DataFrame({
    'nim': ['001', '002', '003', '005'],
    'nilai': [85, 90, 88, 92]
})

# Inner merge: hanya data yang ada di kedua tabel
result = pd.merge(mahasiswa, nilai, on='nim')
print(result)
#    nim   nama  nilai
# 0  001  Ahmad     85
# 1  002   Budi     90
# 2  003  Citra     88
```

### Left Merge

```python
import pandas as pd

mahasiswa = pd.DataFrame({
    'nim': ['001', '002', '003', '004'],
    'nama': ['Ahmad', 'Budi', 'Citra', 'Dani']
})

nilai = pd.DataFrame({
    'nim': ['001', '002', '003', '005'],
    'nilai': [85, 90, 88, 92]
})

# Left merge: semua data dari kiri + matching dari kanan
result = pd.merge(mahasiswa, nilai, on='nim', how='left')
print(result)
#    nim   nama  nilai
# 0  001  Ahmad   85.0
# 1  002   Budi   90.0
# 2  003  Citra   88.0
# 3  004   Dani    NaN
```

### Right Merge

```python
import pandas as pd

mahasiswa = pd.DataFrame({
    'nim': ['001', '002', '003', '004'],
    'nama': ['Ahmad', 'Budi', 'Citra', 'Dani']
})

nilai = pd.DataFrame({
    'nim': ['001', '002', '003', '005'],
    'nilai': [85, 90, 88, 92]
})

# Right merge: semua data dari kanan + matching dari kiri
result = pd.merge(mahasiswa, nilai, on='nim', how='right')
print(result)
#    nim   nama  nilai
# 0  001  Ahmad     85
# 1  002   Budi     90
# 2  003  Citra     88
# 3  005    NaN     92
```

### Outer Merge

```python
import pandas as pd

mahasiswa = pd.DataFrame({
    'nim': ['001', '002', '003', '004'],
    'nama': ['Ahmad', 'Budi', 'Citra', 'Dani']
})

nilai = pd.DataFrame({
    'nim': ['001', '002', '003', '005'],
    'nilai': [85, 90, 88, 92]
})

# Outer merge: semua data dari kedua tabel
result = pd.merge(mahasiswa, nilai, on='nim', how='outer')
print(result)
#    nim   nama  nilai
# 0  001  Ahmad   85.0
# 1  002   Budi   90.0
# 2  003  Citra   88.0
# 3  004   Dani    NaN
# 4  005    NaN   92.0
```

## Merge dengan Nama Kolom Berbeda

```python
import pandas as pd

mahasiswa = pd.DataFrame({
    'nim': ['001', '002', '003'],
    'nama': ['Ahmad', 'Budi', 'Citra']
})

nilai = pd.DataFrame({
    'student_id': ['001', '002', '003'],
    'nilai': [85, 90, 88]
})

# Gunakan left_on dan right_on
result = pd.merge(mahasiswa, nilai, 
                  left_on='nim', 
                  right_on='student_id')
print(result)
#    nim   nama student_id  nilai
# 0  001  Ahmad        001     85
# 1  002   Budi        002     90
# 2  003  Citra        003     88

# Hapus kolom duplikat
result = result.drop('student_id', axis=1)
```

## Merge dengan Multiple Keys

```python
import pandas as pd

df1 = pd.DataFrame({
    'jurusan': ['TI', 'TI', 'SI', 'SI'],
    'semester': [4, 6, 4, 6],
    'jumlah_mahasiswa': [50, 45, 40, 35]
})

df2 = pd.DataFrame({
    'jurusan': ['TI', 'TI', 'SI', 'SI'],
    'semester': [4, 6, 4, 6],
    'rata_ipk': [3.2, 3.4, 3.1, 3.3]
})

result = pd.merge(df1, df2, on=['jurusan', 'semester'])
print(result)
#   jurusan  semester  jumlah_mahasiswa  rata_ipk
# 0      TI         4                50       3.2
# 1      TI         6                45       3.4
# 2      SI         4                40       3.1
# 3      SI         6                35       3.3
```

## Join

Join menggabungkan DataFrame berdasarkan index.

```python
import pandas as pd

df1 = pd.DataFrame({
    'nama': ['Ahmad', 'Budi', 'Citra'],
    'umur': [20, 22, 21]
}, index=['m1', 'm2', 'm3'])

df2 = pd.DataFrame({
    'nilai': [85, 90, 88],
    'status': ['lulus', 'lulus', 'lulus']
}, index=['m1', 'm2', 'm3'])

# Join berdasarkan index
result = df1.join(df2)
print(result)
#      nama  umur  nilai status
# m1  Ahmad    20     85  lulus
# m2   Budi    22     90  lulus
# m3  Citra    21     88  lulus
```

### Join dengan Index Berbeda

```python
import pandas as pd

df1 = pd.DataFrame({
    'nama': ['Ahmad', 'Budi', 'Citra', 'Dani'],
}, index=['m1', 'm2', 'm3', 'm4'])

df2 = pd.DataFrame({
    'nilai': [85, 90, 88],
}, index=['m1', 'm2', 'm5'])

# Left join (default)
print(df1.join(df2))
#      nama  nilai
# m1  Ahmad   85.0
# m2   Budi   90.0
# m3  Citra    NaN
# m4   Dani    NaN

# Outer join
print(df1.join(df2, how='outer'))
```

## Concatenate

Menggabungkan DataFrame secara vertikal atau horizontal.

### Vertikal (Stack)

```python
import pandas as pd

df1 = pd.DataFrame({
    'nama': ['Ahmad', 'Budi'],
    'nilai': [85, 90]
})

df2 = pd.DataFrame({
    'nama': ['Citra', 'Dani'],
    'nilai': [88, 92]
})

# Stack vertikal
result = pd.concat([df1, df2])
print(result)
#     nama  nilai
# 0  Ahmad     85
# 1   Budi     90
# 0  Citra     88
# 1   Dani     92

# Reset index
result = pd.concat([df1, df2], ignore_index=True)
print(result)
#     nama  nilai
# 0  Ahmad     85
# 1   Budi     90
# 2  Citra     88
# 3   Dani     92
```

### Horizontal (Side by Side)

```python
import pandas as pd

df1 = pd.DataFrame({
    'nama': ['Ahmad', 'Budi', 'Citra'],
    'umur': [20, 22, 21]
})

df2 = pd.DataFrame({
    'nilai': [85, 90, 88],
    'status': ['lulus', 'lulus', 'lulus']
})

# Gabung horizontal
result = pd.concat([df1, df2], axis=1)
print(result)
#     nama  umur  nilai status
# 0  Ahmad    20     85  lulus
# 1   Budi    22     90  lulus
# 2  Citra    21     88  lulus
```

### Concat dengan Keys

```python
import pandas as pd

df1 = pd.DataFrame({'nilai': [85, 90]}, index=['Ahmad', 'Budi'])
df2 = pd.DataFrame({'nilai': [88, 92]}, index=['Citra', 'Dani'])

result = pd.concat([df1, df2], keys=['Kelas A', 'Kelas B'])
print(result)
#                nilai
# Kelas A Ahmad     85
#         Budi      90
# Kelas B Citra     88
#         Dani      92
```

## Contoh Praktis

### Menggabungkan Data dari Multiple Sumber

```python
import pandas as pd

# Data mahasiswa
mahasiswa = pd.DataFrame({
    'nim': ['001', '002', '003', '004'],
    'nama': ['Ahmad', 'Budi', 'Citra', 'Dani'],
    'jurusan_id': [1, 2, 1, 2]
})

# Data jurusan
jurusan = pd.DataFrame({
    'id': [1, 2, 3],
    'nama_jurusan': ['Teknik Informatika', 'Sistem Informasi', 'Teknik Komputer']
})

# Data nilai
nilai = pd.DataFrame({
    'nim': ['001', '002', '003', '004'],
    'nilai_uts': [80, 85, 90, 78],
    'nilai_uas': [85, 88, 92, 80]
})

# Gabungkan semua
result = mahasiswa.merge(
    jurusan, 
    left_on='jurusan_id', 
    right_on='id'
).merge(
    nilai, 
    on='nim'
)

# Bersihkan kolom
result = result.drop(['jurusan_id', 'id'], axis=1)
print(result)
```

### Validasi Merge

```python
import pandas as pd

df1 = pd.DataFrame({
    'key': ['A', 'B', 'C'],
    'val1': [1, 2, 3]
})

df2 = pd.DataFrame({
    'key': ['A', 'B', 'B', 'C'],  # B duplikat
    'val2': [4, 5, 6, 7]
})

# Validasi one-to-one
try:
    result = pd.merge(df1, df2, on='key', validate='one_to_one')
except pd.errors.MergeError as e:
    print(f"Error: {e}")

# one-to-many OK
result = pd.merge(df1, df2, on='key', validate='one_to_many')
print(result)
```

## Latihan

1. Gabungkan tabel mahasiswa, jurusan, dan dosen pembimbing
2. Buat laporan lengkap dengan data dari 3 tabel berbeda
3. Concat data nilai semester 1, 2, dan 3 menjadi satu DataFrame
4. Identifikasi mahasiswa yang tidak ada di tabel nilai (left join)
