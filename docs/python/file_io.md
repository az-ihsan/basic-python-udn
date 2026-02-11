# File I/O

Python menyediakan cara mudah untuk membaca dan menulis file. Bagian ini membahas operasi file dasar dan format file umum.

## Membuka File

Gunakan fungsi `open()` untuk membuka file:

```python
# Mode dasar:
# 'r' - Read (baca, default)
# 'w' - Write (tulis, menimpa jika ada)
# 'a' - Append (tambah di akhir)
# 'x' - Create (buat baru, error jika sudah ada)
# 'b' - Binary mode
# 't' - Text mode (default)

file = open("contoh.txt", "r")
# ... operasi file
file.close()
```

## Context Manager (with statement)

Cara yang lebih aman untuk menangani file:

```python
with open("contoh.txt", "r") as file:
    konten = file.read()
    print(konten)
# File otomatis ditutup setelah blok with
```

## Membaca File

### read()

Membaca seluruh isi file:

```python
with open("data.txt", "r") as f:
    konten = f.read()
    print(konten)
```

### readline()

Membaca satu baris:

```python
with open("data.txt", "r") as f:
    baris_pertama = f.readline()
    baris_kedua = f.readline()
    print(baris_pertama)
    print(baris_kedua)
```

### readlines()

Membaca semua baris sebagai list:

```python
with open("data.txt", "r") as f:
    baris = f.readlines()
    for b in baris:
        print(b.strip())  # strip() menghapus newline
```

### Iterasi Langsung

Cara paling efisien untuk file besar:

```python
with open("data.txt", "r") as f:
    for baris in f:
        print(baris.strip())
```

## Menulis File

### write()

Menulis string ke file:

```python
with open("output.txt", "w") as f:
    f.write("Baris pertama\n")
    f.write("Baris kedua\n")
```

### writelines()

Menulis list string:

```python
baris = ["Satu\n", "Dua\n", "Tiga\n"]
with open("output.txt", "w") as f:
    f.writelines(baris)
```

### Mode Append

Menambahkan di akhir file tanpa menimpa:

```python
with open("log.txt", "a") as f:
    f.write("Entry baru\n")
```

## File Binary

Untuk gambar, audio, dan file non-teks lainnya:

```python
# Membaca file binary
with open("gambar.png", "rb") as f:
    data = f.read()

# Menulis file binary
with open("copy.png", "wb") as f:
    f.write(data)
```

## Posisi Pointer File

```python
with open("data.txt", "r") as f:
    # Posisi saat ini
    print(f.tell())  # 0
    
    # Baca 10 karakter
    data = f.read(10)
    print(f.tell())  # 10
    
    # Pindah ke posisi tertentu
    f.seek(0)  # Kembali ke awal
    print(f.tell())  # 0
```

## Encoding

Untuk menangani karakter non-ASCII:

```python
# Menulis dengan encoding UTF-8
with open("data.txt", "w", encoding="utf-8") as f:
    f.write("Halo, Dunia! こんにちは")

# Membaca dengan encoding
with open("data.txt", "r", encoding="utf-8") as f:
    print(f.read())
```

## Bekerja dengan CSV

### Modul csv

```python
import csv

# Menulis CSV
data = [
    ["Nama", "Umur", "Kota"],
    ["Ahmad", 25, "Jakarta"],
    ["Budi", 30, "Bandung"],
    ["Citra", 28, "Surabaya"]
]

with open("data.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerows(data)

# Membaca CSV
with open("data.csv", "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    for baris in reader:
        print(baris)
```

### DictReader dan DictWriter

```python
import csv

# Menulis dengan header
data = [
    {"nama": "Ahmad", "umur": 25, "kota": "Jakarta"},
    {"nama": "Budi", "umur": 30, "kota": "Bandung"}
]

with open("data.csv", "w", newline="", encoding="utf-8") as f:
    fieldnames = ["nama", "umur", "kota"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(data)

# Membaca sebagai dictionary
with open("data.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        print(f"{row['nama']}: {row['umur']} tahun")
```

## Bekerja dengan JSON

```python
import json

# Data Python
data = {
    "mahasiswa": [
        {"nama": "Ahmad", "nilai": 85},
        {"nama": "Budi", "nilai": 90}
    ],
    "mata_kuliah": "Kecerdasan Buatan"
}

# Menulis JSON
with open("data.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

# Membaca JSON
with open("data.json", "r", encoding="utf-8") as f:
    loaded = json.load(f)
    print(loaded["mata_kuliah"])
    for m in loaded["mahasiswa"]:
        print(f"{m['nama']}: {m['nilai']}")
```

## Operasi File dengan os dan pathlib

### Memeriksa Keberadaan File

```python
import os
from pathlib import Path

# Dengan os
print(os.path.exists("data.txt"))
print(os.path.isfile("data.txt"))
print(os.path.isdir("folder"))

# Dengan pathlib
p = Path("data.txt")
print(p.exists())
print(p.is_file())
```

### Mendapatkan Informasi File

```python
import os
from pathlib import Path

# Ukuran file
print(os.path.getsize("data.txt"))

# Dengan pathlib
p = Path("data.txt")
print(p.stat().st_size)
```

### Menghapus dan Memindahkan File

```python
import os
import shutil

# Menghapus file
os.remove("file.txt")

# Menghapus direktori kosong
os.rmdir("folder_kosong")

# Menghapus direktori beserta isinya
shutil.rmtree("folder")

# Memindahkan/rename file
os.rename("lama.txt", "baru.txt")

# Copy file
shutil.copy("source.txt", "destination.txt")

# Copy direktori
shutil.copytree("source_folder", "dest_folder")
```

## Contoh Praktis

### Log File Sederhana

```python
from datetime import datetime

def log(pesan):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("app.log", "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {pesan}\n")

log("Aplikasi dimulai")
log("User login: Ahmad")
log("Proses selesai")
```

### Membaca Konfigurasi

```python
import json

def load_config(filename="config.json"):
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"debug": False, "port": 8080}

def save_config(config, filename="config.json"):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

# Penggunaan
config = load_config()
config["debug"] = True
save_config(config)
```

### Menggabungkan File Teks

```python
from pathlib import Path

def gabung_file(folder, output):
    files = sorted(Path(folder).glob("*.txt"))
    with open(output, "w", encoding="utf-8") as out:
        for f in files:
            out.write(f"--- {f.name} ---\n")
            out.write(f.read_text(encoding="utf-8"))
            out.write("\n\n")

gabung_file("dokumen/", "gabungan.txt")
```

## Latihan

1. Buat program untuk menghitung jumlah kata dalam file teks
2. Buat program untuk membaca CSV nilai mahasiswa dan menghitung rata-rata
3. Buat program sederhana untuk menyimpan dan membaca to-do list dari file JSON
4. Buat program untuk backup file dengan menambahkan timestamp pada nama file
