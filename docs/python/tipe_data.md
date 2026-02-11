# Tipe Data

Python memiliki beberapa tipe data bawaan yang penting untuk dipahami. Mari kita pelajari satu per satu.

## Tipe Data Numerik

### Integer (int)

Bilangan bulat tanpa desimal:

```python
x = 10
y = -5
z = 0

print(type(x))  # <class 'int'>
```

### Float

Bilangan dengan desimal:

```python
pi = 3.14159
suhu = -273.15

print(type(pi))  # <class 'float'>
```

### Operasi Aritmatika

```python
a = 10
b = 3

print(a + b)   # 13  - Penjumlahan
print(a - b)   # 7   - Pengurangan
print(a * b)   # 30  - Perkalian
print(a / b)   # 3.333... - Pembagian (hasil float)
print(a // b)  # 3   - Pembagian bulat
print(a % b)   # 1   - Modulo (sisa bagi)
print(a ** b)  # 1000 - Pangkat
```

## String (str)

Teks atau rangkaian karakter:

```python
nama = "Ahmad"
pesan = 'Selamat datang!'
paragraf = """Ini adalah
teks multi-baris"""

print(type(nama))  # <class 'str'>
```

### Operasi String

```python
# Concatenation (penggabungan)
sapaan = "Halo, " + "Dunia!"
print(sapaan)  # Halo, Dunia!

# Repetisi
garis = "-" * 20
print(garis)  # --------------------

# Panjang string
print(len(nama))  # 5

# Indexing (dimulai dari 0)
print(nama[0])   # A
print(nama[-1])  # d (dari belakang)

# Slicing
print(nama[0:3])  # Ahm
print(nama[2:])   # mad
print(nama[:3])   # Ahm
```

### String Methods

```python
teks = "  Halo Dunia  "

print(teks.upper())       # "  HALO DUNIA  "
print(teks.lower())       # "  halo dunia  "
print(teks.strip())       # "Halo Dunia" (hapus spasi)
print(teks.replace("Halo", "Hai"))  # "  Hai Dunia  "
print(teks.split())       # ['Halo', 'Dunia']
```

### F-String (Formatted String)

Cara modern untuk memformat string:

```python
nama = "Ahmad"
umur = 25

# F-string
pesan = f"Nama saya {nama}, umur {umur} tahun"
print(pesan)  # Nama saya Ahmad, umur 25 tahun

# Ekspresi dalam f-string
print(f"2 + 3 = {2 + 3}")  # 2 + 3 = 5

# Format angka
pi = 3.14159
print(f"Pi = {pi:.2f}")  # Pi = 3.14
```

## Boolean (bool)

Nilai kebenaran True atau False:

```python
aktif = True
selesai = False

print(type(aktif))  # <class 'bool'>
```

### Operator Perbandingan

```python
x = 10
y = 5

print(x == y)   # False - sama dengan
print(x != y)   # True  - tidak sama dengan
print(x > y)    # True  - lebih besar
print(x < y)    # False - lebih kecil
print(x >= y)   # True  - lebih besar atau sama
print(x <= y)   # False - lebih kecil atau sama
```

### Operator Logika

```python
a = True
b = False

print(a and b)  # False - DAN
print(a or b)   # True  - ATAU
print(not a)    # False - NEGASI
```

## List

Koleksi terurut yang dapat diubah (mutable):

```python
angka = [1, 2, 3, 4, 5]
campuran = [1, "dua", 3.0, True]

print(type(angka))  # <class 'list'>
```

### Operasi List

```python
buah = ["apel", "jeruk", "mangga"]

# Akses elemen
print(buah[0])   # apel
print(buah[-1])  # mangga

# Modifikasi
buah[1] = "pisang"
print(buah)  # ['apel', 'pisang', 'mangga']

# Menambah elemen
buah.append("durian")
print(buah)  # ['apel', 'pisang', 'mangga', 'durian']

# Menghapus elemen
buah.remove("pisang")
print(buah)  # ['apel', 'mangga', 'durian']

# Panjang list
print(len(buah))  # 3

# Slicing
print(buah[0:2])  # ['apel', 'mangga']
```

### List Comprehension

Cara ringkas membuat list:

```python
# Cara biasa
kuadrat = []
for i in range(5):
    kuadrat.append(i ** 2)

# List comprehension
kuadrat = [i ** 2 for i in range(5)]
print(kuadrat)  # [0, 1, 4, 9, 16]

# Dengan kondisi
genap = [i for i in range(10) if i % 2 == 0]
print(genap)  # [0, 2, 4, 6, 8]
```

## Tuple

Koleksi terurut yang tidak dapat diubah (immutable):

```python
koordinat = (10, 20)
rgb = (255, 128, 0)

print(type(koordinat))  # <class 'tuple'>

# Akses elemen
print(koordinat[0])  # 10

# Tuple tidak bisa diubah
# koordinat[0] = 5  # Error!
```

### Tuple Unpacking

```python
koordinat = (10, 20, 30)
x, y, z = koordinat

print(x)  # 10
print(y)  # 20
print(z)  # 30
```

## Dictionary (dict)

Koleksi pasangan kunci-nilai:

```python
mahasiswa = {
    "nama": "Ahmad",
    "nim": "12345",
    "jurusan": "Informatika"
}

print(type(mahasiswa))  # <class 'dict'>
```

### Operasi Dictionary

```python
# Akses nilai
print(mahasiswa["nama"])      # Ahmad
print(mahasiswa.get("nim"))   # 12345
print(mahasiswa.get("umur", 0))  # 0 (default jika tidak ada)

# Modifikasi
mahasiswa["umur"] = 20
print(mahasiswa)

# Menambah pasangan baru
mahasiswa["email"] = "ahmad@udn.ac.id"

# Menghapus
del mahasiswa["email"]

# Iterasi
for kunci in mahasiswa:
    print(f"{kunci}: {mahasiswa[kunci]}")

# Atau
for kunci, nilai in mahasiswa.items():
    print(f"{kunci}: {nilai}")
```

## Set

Koleksi tanpa urutan dan tanpa duplikat:

```python
unik = {1, 2, 3, 3, 2, 1}
print(unik)  # {1, 2, 3}

print(type(unik))  # <class 'set'>
```

### Operasi Set

```python
a = {1, 2, 3, 4}
b = {3, 4, 5, 6}

print(a | b)  # {1, 2, 3, 4, 5, 6} - Union
print(a & b)  # {3, 4} - Intersection
print(a - b)  # {1, 2} - Difference
print(a ^ b)  # {1, 2, 5, 6} - Symmetric Difference
```

## None

Nilai kosong atau tidak ada:

```python
x = None
print(x)         # None
print(type(x))   # <class 'NoneType'>

# Mengecek None
if x is None:
    print("x adalah None")
```

## Konversi Tipe Data

```python
# String ke integer
x = int("10")
print(x, type(x))  # 10 <class 'int'>

# Integer ke string
y = str(25)
print(y, type(y))  # 25 <class 'str'>

# String ke float
z = float("3.14")
print(z, type(z))  # 3.14 <class 'float'>

# List ke tuple
daftar = [1, 2, 3]
t = tuple(daftar)
print(t)  # (1, 2, 3)
```

## Ringkasan

| Tipe Data | Contoh | Mutable |
|-----------|--------|---------|
| int | `10`, `-5` | Tidak |
| float | `3.14`, `-0.5` | Tidak |
| str | `"Halo"` | Tidak |
| bool | `True`, `False` | Tidak |
| list | `[1, 2, 3]` | Ya |
| tuple | `(1, 2, 3)` | Tidak |
| dict | `{"a": 1}` | Ya |
| set | `{1, 2, 3}` | Ya |
| None | `None` | - |

## Latihan

1. Buat variabel untuk menyimpan nama, umur, dan tinggi badan Anda
2. Buat list berisi 5 buah favorit Anda
3. Buat dictionary untuk menyimpan data mahasiswa (nama, nim, jurusan, IPK)
4. Gunakan f-string untuk mencetak informasi dari dictionary tersebut
