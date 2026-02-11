# Error Handling

Error handling memungkinkan program menangani kesalahan dengan anggun tanpa crash. Python menggunakan mekanisme try-except untuk menangani exception.

## Jenis Error

### Syntax Error

Kesalahan penulisan kode yang terdeteksi saat parsing:

```python
# SyntaxError - tanda kurung tidak lengkap
# print("Halo"

# SyntaxError - indentasi salah
# if True:
# print("test")
```

### Exception

Kesalahan yang terjadi saat runtime:

```python
# ZeroDivisionError
# hasil = 10 / 0

# TypeError
# hasil = "2" + 2

# NameError
# print(variabel_tidak_ada)

# IndexError
# list = [1, 2, 3]
# print(list[10])

# KeyError
# dict = {"a": 1}
# print(dict["b"])

# FileNotFoundError
# f = open("tidak_ada.txt")
```

## Try-Except

Menangkap dan menangani exception:

```python
try:
    hasil = 10 / 0
except ZeroDivisionError:
    print("Error: Tidak bisa membagi dengan nol!")

print("Program berlanjut...")
```

### Menangkap Multiple Exceptions

```python
try:
    angka = int(input("Masukkan angka: "))
    hasil = 10 / angka
except ValueError:
    print("Error: Input bukan angka!")
except ZeroDivisionError:
    print("Error: Tidak bisa membagi dengan nol!")
```

### Menangkap Exception dalam Satu Blok

```python
try:
    # kode yang mungkin error
    pass
except (ValueError, TypeError) as e:
    print(f"Error: {e}")
```

### Menangkap Semua Exception

```python
try:
    # kode berbahaya
    pass
except Exception as e:
    print(f"Terjadi error: {e}")
```

:::{warning}
Menangkap semua exception sebaiknya dihindari karena bisa menyembunyikan bug yang tidak terduga. Gunakan hanya jika benar-benar diperlukan.
:::

## Else dan Finally

### else

Dieksekusi jika tidak ada exception:

```python
try:
    angka = int("42")
except ValueError:
    print("Bukan angka!")
else:
    print(f"Berhasil: {angka}")
```

### finally

Selalu dieksekusi, ada exception atau tidak:

```python
try:
    f = open("data.txt", "r")
    data = f.read()
except FileNotFoundError:
    print("File tidak ditemukan!")
finally:
    print("Blok finally selalu dieksekusi")
    # Pastikan file ditutup
```

### Contoh Lengkap

```python
def baca_file(nama_file):
    try:
        f = open(nama_file, "r")
        data = f.read()
    except FileNotFoundError:
        print(f"File {nama_file} tidak ditemukan!")
        return None
    except PermissionError:
        print("Tidak ada izin untuk membaca file!")
        return None
    else:
        print("File berhasil dibaca!")
        return data
    finally:
        print("Operasi selesai")
        try:
            f.close()
        except:
            pass
```

## Raise Exception

Membangkitkan exception secara manual:

```python
def set_umur(umur):
    if umur < 0:
        raise ValueError("Umur tidak boleh negatif!")
    if umur > 150:
        raise ValueError("Umur tidak realistis!")
    return umur

try:
    set_umur(-5)
except ValueError as e:
    print(f"Error: {e}")
```

## Custom Exception

Membuat exception sendiri:

```python
class NilaiTidakValidError(Exception):
    """Exception untuk nilai yang tidak valid."""
    pass

class NilaiTerlaluTinggiError(NilaiTidakValidError):
    """Exception untuk nilai lebih dari 100."""
    pass

class NilaiNegatifError(NilaiTidakValidError):
    """Exception untuk nilai negatif."""
    pass

def set_nilai(nilai):
    if nilai < 0:
        raise NilaiNegatifError(f"Nilai {nilai} tidak boleh negatif!")
    if nilai > 100:
        raise NilaiTerlaluTinggiError(f"Nilai {nilai} melebihi 100!")
    return nilai

try:
    set_nilai(150)
except NilaiTerlaluTinggiError as e:
    print(f"Error nilai terlalu tinggi: {e}")
except NilaiNegatifError as e:
    print(f"Error nilai negatif: {e}")
except NilaiTidakValidError as e:
    print(f"Error nilai tidak valid: {e}")
```

## Assert

Memeriksa kondisi dan raise AssertionError jika False:

```python
def hitung_rata_rata(nilai):
    assert len(nilai) > 0, "List tidak boleh kosong!"
    return sum(nilai) / len(nilai)

try:
    rata = hitung_rata_rata([])
except AssertionError as e:
    print(f"Assertion gagal: {e}")
```

:::{note}
Assert sebaiknya digunakan untuk debugging, bukan untuk menangani input user. Assert bisa dinonaktifkan dengan flag `-O`.
:::

## Exception Chaining

Menghubungkan exception dengan exception lain:

```python
def proses_data(filename):
    try:
        f = open(filename)
        data = f.read()
    except FileNotFoundError as e:
        raise ValueError(f"Tidak bisa memproses: file tidak ada") from e

try:
    proses_data("tidak_ada.txt")
except ValueError as e:
    print(f"Error: {e}")
    print(f"Penyebab: {e.__cause__}")
```

## Context Manager untuk Error Handling

Menggunakan contextlib untuk penanganan yang lebih bersih:

```python
from contextlib import contextmanager

@contextmanager
def buka_file(nama):
    f = None
    try:
        f = open(nama, "r")
        yield f
    except FileNotFoundError:
        print(f"File {nama} tidak ditemukan!")
        yield None
    finally:
        if f:
            f.close()

with buka_file("data.txt") as f:
    if f:
        print(f.read())
```

## Logging Error

Menggunakan modul logging untuk mencatat error:

```python
import logging

# Konfigurasi logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="app.log"
)

def bagi(a, b):
    try:
        hasil = a / b
        logging.info(f"Berhasil: {a} / {b} = {hasil}")
        return hasil
    except ZeroDivisionError as e:
        logging.error(f"Error pembagian: {e}")
        raise
    except TypeError as e:
        logging.error(f"Error tipe data: {e}")
        raise

# Penggunaan
try:
    bagi(10, 0)
except ZeroDivisionError:
    print("Terjadi kesalahan!")
```

## Best Practices

### 1. Spesifik dalam Menangkap Exception

```python
# Buruk - terlalu umum
try:
    ...
except:
    pass

# Baik - spesifik
try:
    ...
except ValueError:
    ...
except TypeError:
    ...
```

### 2. Jangan Mengabaikan Exception

```python
# Buruk
try:
    operasi_penting()
except:
    pass

# Baik
try:
    operasi_penting()
except SomeError as e:
    logging.error(f"Gagal: {e}")
    # Atau re-raise
    raise
```

### 3. Gunakan finally untuk Cleanup

```python
resource = acquire_resource()
try:
    use_resource(resource)
finally:
    release_resource(resource)

# Atau lebih baik dengan context manager
with acquire_resource() as resource:
    use_resource(resource)
```

### 4. Berikan Pesan Error yang Jelas

```python
# Buruk
raise ValueError("Error!")

# Baik
raise ValueError(f"Nilai '{nilai}' harus berupa integer positif")
```

## Contoh Praktis

### Validasi Input User

```python
def minta_angka(prompt, min_val=None, max_val=None):
    while True:
        try:
            nilai = float(input(prompt))
            if min_val is not None and nilai < min_val:
                raise ValueError(f"Nilai minimal adalah {min_val}")
            if max_val is not None and nilai > max_val:
                raise ValueError(f"Nilai maksimal adalah {max_val}")
            return nilai
        except ValueError as e:
            print(f"Input tidak valid: {e}")

umur = minta_angka("Masukkan umur: ", min_val=0, max_val=150)
```

### Retry dengan Backoff

```python
import time

def dengan_retry(func, max_attempts=3, delay=1):
    for attempt in range(max_attempts):
        try:
            return func()
        except Exception as e:
            if attempt == max_attempts - 1:
                raise
            print(f"Attempt {attempt + 1} gagal: {e}")
            time.sleep(delay * (attempt + 1))

# Penggunaan
def operasi_tidak_stabil():
    import random
    if random.random() < 0.7:
        raise ConnectionError("Koneksi terputus")
    return "Sukses!"

try:
    hasil = dengan_retry(operasi_tidak_stabil)
    print(hasil)
except ConnectionError:
    print("Gagal setelah semua retry")
```

## Latihan

1. Buat fungsi yang membaca file dan menangani berbagai kemungkinan error
2. Buat custom exception untuk validasi data mahasiswa
3. Buat decorator untuk menangani exception dan melakukan logging
4. Implementasikan retry mechanism untuk koneksi database
