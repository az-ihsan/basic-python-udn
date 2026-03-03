# Modul dan Paket

Modul dan paket memungkinkan kita mengatur kode Python menjadi unit terpisah yang dapat digunakan ulang.

## Tujuan Pembelajaran

Setelah mempelajari bab ini, Anda diharapkan mampu:

- Menjelaskan perbedaan modul dan paket
- Menggunakan beberapa gaya `import` dengan tepat
- Memanfaatkan modul standar Python untuk tugas umum
- Mengelola dependensi proyek dengan `pip` dan virtual environment

## Modul

Modul adalah file Python (`.py`) yang berisi definisi fungsi, kelas, dan variabel.

### Membuat Modul

Buat file `matematika.py`:

```python
# matematika.py
PI = 3.14159

def luas_lingkaran(r):
    return PI * r ** 2

def keliling_lingkaran(r):
    return 2 * PI * r

def luas_persegi(sisi):
    return sisi ** 2
```

### Menggunakan Modul

```python
# Cara 1: Import seluruh modul (paling aman untuk pemula)
import matematika

print(matematika.PI)
print(matematika.luas_lingkaran(5))

# Cara 2: Import objek spesifik
from matematika import luas_lingkaran, PI

print(PI)
print(luas_lingkaran(5))

# Cara 3: Import semua (hindari)
from matematika import *

# Cara 4: Import dengan alias
import matematika as mat

print(mat.luas_persegi(4))
```

:::{warning}
Hindari `from modul import *` karena dapat mengotori namespace dan menimbulkan bentrokan nama yang sulit dilacak.
:::

## Modul Standar Python

Python memiliki banyak modul bawaan yang berguna:

### math

```python
import math

print(math.pi)       # 3.141592653589793
print(math.e)        # 2.718281828459045
print(math.sqrt(16)) # 4.0
print(math.sin(0))   # 0.0
print(math.log(10))  # 2.302585...
print(math.ceil(4.2))  # 5
print(math.floor(4.8)) # 4
```

### random

```python
import random

# Bilangan pecahan acak antara 0 dan 1
print(random.random())

# Bilangan bulat acak dalam rentang [1, 10]
print(random.randint(1, 10))

# Pilih elemen acak dari list
buah = ["apel", "jeruk", "mangga"]
print(random.choice(buah))

# Acak urutan list (in-place)
angka = [1, 2, 3, 4, 5]
random.shuffle(angka)
print(angka)

# Ambil sampel unik sebanyak 5 elemen
print(random.sample(range(100), 5))
```

### datetime

```python
from datetime import datetime, date, timedelta

# Waktu sekarang
sekarang = datetime.now()
print(sekarang)

# Tanggal hari ini
hari_ini = date.today()
print(hari_ini)

# Format tanggal
print(sekarang.strftime("%d/%m/%Y %H:%M:%S"))

# Parsing string ke datetime
tgl = datetime.strptime("25/12/2026", "%d/%m/%Y")
print(tgl)

# Operasi dengan timedelta
besok = hari_ini + timedelta(days=1)
minggu_depan = hari_ini + timedelta(weeks=1)
print(f"Besok: {besok}")
print(f"Minggu depan: {minggu_depan}")
```

### os

```python
import os

# Direktori kerja saat ini
print(os.getcwd())

# Daftar file dalam direktori
print(os.listdir("."))

# Membuat direktori
# os.mkdir("folder_baru")

# Cek keberadaan file/folder
print(os.path.exists("file.txt"))

# Mendapatkan nama file dan direktori
path = os.path.join("home", "user", "dokumen", "file.txt")
print(os.path.basename(path))  # nama berkas
print(os.path.dirname(path))   # direktori induk

# Menggabungkan path
full_path = os.path.join("folder", "subfolder", "file.txt")
print(full_path)
```

### pathlib (Penanganan Path Modern)

```python
from pathlib import Path

# Path saat ini
cwd = Path.cwd()
print(cwd)

# Home directory
home = Path.home()
print(home)

# Membuat path
p = Path("folder") / "subfolder" / "file.txt"
print(p)

# Cek keberadaan
print(p.exists())

# Mendapatkan ekstensi
print(Path("data.csv").suffix)  # .csv

# Iterasi file dalam direktori
for file in Path(".").glob("*.py"):
    print(file)
```

### json

```python
import json

# Python dict ke JSON string
data = {"nama": "Ahmad", "umur": 25}
json_str = json.dumps(data, indent=2)
print(json_str)

# JSON string ke Python dict
parsed = json.loads(json_str)
print(parsed["nama"])

# Menulis ke file JSON
with open("data.json", "w") as f:
    json.dump(data, f, indent=2)

# Membaca dari file JSON
with open("data.json", "r") as f:
    loaded = json.load(f)
    print(loaded)
```

### collections

```python
from collections import Counter, defaultdict, namedtuple

# Counter - menghitung frekuensi
kata = ["apel", "jeruk", "apel", "mangga", "apel", "jeruk"]
counter = Counter(kata)
print(counter)  # Counter({'apel': 3, 'jeruk': 2, 'mangga': 1})
print(counter.most_common(2))  # [('apel', 3), ('jeruk', 2)]

# defaultdict - dict dengan nilai default
dd = defaultdict(list)
dd["buah"].append("apel")
dd["buah"].append("jeruk")
print(dd)  # defaultdict(<class 'list'>, {'buah': ['apel', 'jeruk']})

# namedtuple - tuple dengan nama field
Titik = namedtuple("Titik", ["x", "y"])
p = Titik(10, 20)
print(p.x, p.y)  # 10 20
```

### itertools

Modul `itertools` menyediakan fungsi-fungsi untuk bekerja dengan iterator secara efisien, termasuk operasi kombinatorik seperti permutasi, kombinasi, dan Cartesian product.

Untuk materi lengkap tentang permutasi, kombinasi, faktorial, Cartesian product, dan aplikasinya dalam probabilitas dan statistik, lihat:

- [Itertools dan Kombinatorik](itertools_combinatorik.md)

## Paket

Paket adalah direktori yang berisi modul-modul Python.

Dalam praktik modern, paket bisa berupa **namespace package** (tanpa `__init__.py`). Namun, untuk pembelajaran dasar, tetap disarankan memakai `__init__.py` agar struktur paket eksplisit dan mudah dipahami.

### Struktur Paket

```
mypackage/
├── __init__.py
├── modul_a.py
├── modul_b.py
└── subpackage/
    ├── __init__.py
    └── modul_c.py
```

### Membuat Paket

```python
# mypackage/__init__.py
"""Paket utama mypackage."""
from .modul_a import fungsi_a
from .modul_b import fungsi_b

__version__ = "1.0.0"

# mypackage/modul_a.py
def fungsi_a():
    return "Dari modul A"

# mypackage/modul_b.py
def fungsi_b():
    return "Dari modul B"

# mypackage/subpackage/__init__.py
from .modul_c import fungsi_c

# mypackage/subpackage/modul_c.py
def fungsi_c():
    return "Dari modul C di subpackage"
```

### Menggunakan Paket

```python
# Import dari paket
import mypackage
print(mypackage.fungsi_a())

# Import modul spesifik
from mypackage import modul_a
print(modul_a.fungsi_a())

# Import dari subpackage
from mypackage.subpackage import modul_c
print(modul_c.fungsi_c())

# Import langsung
from mypackage.subpackage.modul_c import fungsi_c
print(fungsi_c())
```

## Instalasi Paket Eksternal

### pip

```bash
# Instal paket
python -m pip install numpy

# Instal versi spesifik
python -m pip install numpy==1.24.0

# Instal dari requirements.txt
python -m pip install -r requirements.txt

# Lihat paket terinstal
python -m pip list

# Copot paket
python -m pip uninstall numpy
```

### requirements.txt

```
# requirements.txt
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
torch>=2.0.0
```

## Virtual Environment dan Dependensi

### Membuat Virtual Environment

```bash
# Buat virtual environment
python -m venv .venv

# Aktivasi
source .venv/bin/activate  # Linux/macOS
.\.venv\Scripts\activate   # Windows (PowerShell)

# Instal dependensi
python -m pip install -r requirements.txt

# Ekspor dependensi terpasang
python -m pip freeze > requirements.txt
```

:::{note}
`pip freeze` cocok untuk mengunci dependensi lingkungan proyek saat ini. Untuk modul ajar atau pustaka, gunakan pin versi secara sadar (tidak selalu semua paket perlu dibekukan).
:::

## if __name__ == "__main__"

Idiom untuk menjalankan kode hanya jika file dieksekusi langsung:

```python
# utils.py
def fungsi_helper():
    return "Helper"

def main():
    print("Program utama")
    print(fungsi_helper())

if __name__ == "__main__":
    main()
```

- Jika dijalankan langsung: `python utils.py` → main() dieksekusi
- Jika diimpor: `import utils` → `main()` tidak dieksekusi

## Latihan

1. Buat modul `geometri.py` dengan fungsi untuk menghitung luas berbagai bangun datar
2. Buat paket `tools` dengan submodul `text` dan `math`
3. Gunakan modul `random` untuk membuat program undian sederhana
4. Eksplorasi modul `datetime` untuk membuat kalkulator usia
