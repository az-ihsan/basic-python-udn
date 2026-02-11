# Fungsi

Fungsi adalah blok kode yang dapat digunakan kembali untuk melakukan tugas tertentu. Fungsi membantu mengorganisir kode dan menghindari pengulangan.

## Mendefinisikan Fungsi

```python
def sapa():
    print("Halo, Dunia!")

# Memanggil fungsi
sapa()  # Output: Halo, Dunia!
```

## Parameter dan Argumen

### Parameter Posisional

```python
def sapa(nama):
    print(f"Halo, {nama}!")

sapa("Ahmad")  # Halo, Ahmad!
sapa("Budi")   # Halo, Budi!
```

### Multiple Parameters

```python
def perkenalan(nama, umur, kota):
    print(f"Nama: {nama}")
    print(f"Umur: {umur}")
    print(f"Kota: {kota}")

perkenalan("Ahmad", 25, "Jakarta")
```

### Parameter Default

```python
def sapa(nama, sapaan="Halo"):
    print(f"{sapaan}, {nama}!")

sapa("Ahmad")              # Halo, Ahmad!
sapa("Budi", "Selamat pagi")  # Selamat pagi, Budi!
```

### Keyword Arguments

```python
def profil(nama, umur, pekerjaan):
    print(f"{nama}, {umur} tahun, {pekerjaan}")

# Menggunakan keyword arguments
profil(umur=30, nama="Ahmad", pekerjaan="Programmer")
```

## Return Value

Fungsi dapat mengembalikan nilai dengan `return`:

```python
def tambah(a, b):
    return a + b

hasil = tambah(5, 3)
print(hasil)  # 8
```

### Multiple Return Values

```python
def statistik(angka):
    minimum = min(angka)
    maksimum = max(angka)
    rata_rata = sum(angka) / len(angka)
    return minimum, maksimum, rata_rata

data = [10, 20, 30, 40, 50]
min_val, max_val, avg = statistik(data)

print(f"Min: {min_val}")  # Min: 10
print(f"Max: {max_val}")  # Max: 50
print(f"Avg: {avg}")      # Avg: 30.0
```

### Return None

Fungsi tanpa return eksplisit mengembalikan None:

```python
def cetak_pesan(pesan):
    print(pesan)

hasil = cetak_pesan("Halo")
print(hasil)  # None
```

## *args dan **kwargs

### *args (Arbitrary Positional Arguments)

Menerima jumlah argumen yang tidak terbatas:

```python
def jumlah(*args):
    total = 0
    for angka in args:
        total += angka
    return total

print(jumlah(1, 2, 3))        # 6
print(jumlah(1, 2, 3, 4, 5))  # 15
```

### **kwargs (Arbitrary Keyword Arguments)

Menerima keyword arguments yang tidak terbatas:

```python
def profil(**kwargs):
    for kunci, nilai in kwargs.items():
        print(f"{kunci}: {nilai}")

profil(nama="Ahmad", umur=25, kota="Jakarta")
# Output:
# nama: Ahmad
# umur: 25
# kota: Jakarta
```

### Kombinasi

```python
def fungsi(a, b, *args, **kwargs):
    print(f"a = {a}")
    print(f"b = {b}")
    print(f"args = {args}")
    print(f"kwargs = {kwargs}")

fungsi(1, 2, 3, 4, 5, nama="Ahmad", umur=25)
# a = 1
# b = 2
# args = (3, 4, 5)
# kwargs = {'nama': 'Ahmad', 'umur': 25}
```

## Docstrings

Dokumentasi fungsi menggunakan docstring:

```python
def hitung_luas_persegi(sisi):
    """
    Menghitung luas persegi.
    
    Parameters
    ----------
    sisi : float
        Panjang sisi persegi
        
    Returns
    -------
    float
        Luas persegi
        
    Examples
    --------
    >>> hitung_luas_persegi(5)
    25
    """
    return sisi ** 2

# Mengakses docstring
print(hitung_luas_persegi.__doc__)
help(hitung_luas_persegi)
```

## Lambda Functions

Fungsi anonim satu baris:

```python
# Fungsi biasa
def kuadrat(x):
    return x ** 2

# Equivalent lambda
kuadrat = lambda x: x ** 2

print(kuadrat(5))  # 25
```

### Lambda dengan Multiple Parameters

```python
tambah = lambda a, b: a + b
print(tambah(3, 4))  # 7

# Dengan sorted
mahasiswa = [
    {"nama": "Ahmad", "nilai": 85},
    {"nama": "Budi", "nilai": 90},
    {"nama": "Citra", "nilai": 78}
]

# Sortir berdasarkan nilai
terurut = sorted(mahasiswa, key=lambda x: x["nilai"], reverse=True)
for m in terurut:
    print(f"{m['nama']}: {m['nilai']}")
```

## Higher-Order Functions

### map()

Menerapkan fungsi ke setiap elemen:

```python
angka = [1, 2, 3, 4, 5]
kuadrat = list(map(lambda x: x ** 2, angka))
print(kuadrat)  # [1, 4, 9, 16, 25]
```

### filter()

Menyaring elemen berdasarkan kondisi:

```python
angka = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
genap = list(filter(lambda x: x % 2 == 0, angka))
print(genap)  # [2, 4, 6, 8, 10]
```

### reduce()

Mengurangi list menjadi satu nilai:

```python
from functools import reduce

angka = [1, 2, 3, 4, 5]
jumlah = reduce(lambda a, b: a + b, angka)
print(jumlah)  # 15
```

## Scope Variabel

### Local dan Global Scope

```python
x = 10  # Global

def fungsi():
    y = 5  # Local
    print(f"x (global): {x}")
    print(f"y (local): {y}")

fungsi()
# print(y)  # Error: y tidak ada di global scope
```

### global Keyword

```python
counter = 0

def increment():
    global counter
    counter += 1

increment()
increment()
print(counter)  # 2
```

### nonlocal Keyword

Untuk nested functions:

```python
def outer():
    x = 10
    
    def inner():
        nonlocal x
        x += 5
    
    inner()
    print(x)  # 15

outer()
```

## Recursion

Fungsi yang memanggil dirinya sendiri:

```python
def faktorial(n):
    if n <= 1:
        return 1
    return n * faktorial(n - 1)

print(faktorial(5))  # 120
```

### Fibonacci Recursive

```python
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

# Cetak 10 angka Fibonacci pertama
for i in range(10):
    print(fibonacci(i), end=" ")
# 0 1 1 2 3 5 8 13 21 34
```

## Type Hints (Python 3.5+)

Menambahkan tipe data pada parameter dan return:

```python
def sapa(nama: str) -> str:
    return f"Halo, {nama}!"

def tambah(a: int, b: int) -> int:
    return a + b

def proses_data(data: list[int]) -> dict[str, float]:
    return {
        "min": min(data),
        "max": max(data),
        "avg": sum(data) / len(data)
    }
```

## Decorators

Fungsi yang memodifikasi fungsi lain:

```python
def uppercase_decorator(func):
    def wrapper(*args, **kwargs):
        hasil = func(*args, **kwargs)
        return hasil.upper()
    return wrapper

@uppercase_decorator
def sapa(nama):
    return f"Halo, {nama}"

print(sapa("ahmad"))  # HALO, AHMAD
```

### Decorator dengan Timing

```python
import time

def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        hasil = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} selesai dalam {end - start:.4f} detik")
        return hasil
    return wrapper

@timer
def operasi_lambat():
    time.sleep(1)
    return "Selesai"

operasi_lambat()
# operasi_lambat selesai dalam 1.0012 detik
```

## Latihan

1. Buat fungsi untuk menghitung luas dan keliling lingkaran
2. Buat fungsi rekursif untuk mencari nilai maksimum dalam list
3. Buat decorator untuk meng-cache hasil fungsi (memoization)
4. Buat fungsi yang menerima *args dan mengembalikan rata-rata
