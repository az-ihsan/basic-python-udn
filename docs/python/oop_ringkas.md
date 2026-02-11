# OOP Ringkas

Object-Oriented Programming (OOP) adalah paradigma pemrograman yang menggunakan "objek" untuk memodelkan data dan perilaku. Bagian ini membahas konsep OOP dasar di Python.

## Class dan Object

**Class** adalah blueprint untuk membuat object. **Object** adalah instance dari class.

```python
# Mendefinisikan class
class Mahasiswa:
    pass

# Membuat object (instance)
mhs1 = Mahasiswa()
mhs2 = Mahasiswa()

print(type(mhs1))  # <class '__main__.Mahasiswa'>
```

## Atribut dan Method

### Instance Attributes

```python
class Mahasiswa:
    def __init__(self, nama, nim):
        self.nama = nama  # instance attribute
        self.nim = nim

# Membuat object dengan atribut
mhs = Mahasiswa("Ahmad", "12345")
print(mhs.nama)  # Ahmad
print(mhs.nim)   # 12345
```

### Class Attributes

Atribut yang dibagi oleh semua instance:

```python
class Mahasiswa:
    universitas = "UDN"  # class attribute
    
    def __init__(self, nama, nim):
        self.nama = nama
        self.nim = nim

mhs1 = Mahasiswa("Ahmad", "12345")
mhs2 = Mahasiswa("Budi", "12346")

print(mhs1.universitas)  # UDN
print(mhs2.universitas)  # UDN
print(Mahasiswa.universitas)  # UDN
```

### Methods

Fungsi di dalam class:

```python
class Mahasiswa:
    def __init__(self, nama, nim):
        self.nama = nama
        self.nim = nim
        self.nilai = []
    
    def tambah_nilai(self, nilai):
        self.nilai.append(nilai)
    
    def rata_rata(self):
        if not self.nilai:
            return 0
        return sum(self.nilai) / len(self.nilai)
    
    def info(self):
        return f"{self.nama} ({self.nim})"

mhs = Mahasiswa("Ahmad", "12345")
mhs.tambah_nilai(85)
mhs.tambah_nilai(90)
print(mhs.info())        # Ahmad (12345)
print(mhs.rata_rata())   # 87.5
```

## Special Methods (Dunder Methods)

Method khusus dengan double underscore:

```python
class Titik:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __str__(self):
        """Representasi string untuk print()"""
        return f"Titik({self.x}, {self.y})"
    
    def __repr__(self):
        """Representasi untuk debugging"""
        return f"Titik(x={self.x}, y={self.y})"
    
    def __eq__(self, other):
        """Perbandingan kesamaan (==)"""
        return self.x == other.x and self.y == other.y
    
    def __add__(self, other):
        """Operator penjumlahan (+)"""
        return Titik(self.x + other.x, self.y + other.y)

p1 = Titik(1, 2)
p2 = Titik(3, 4)

print(p1)           # Titik(1, 2)
print(p1 + p2)      # Titik(4, 6)
print(p1 == p2)     # False
```

## Inheritance (Pewarisan)

Class dapat mewarisi atribut dan method dari class lain:

```python
# Parent class (superclass)
class Orang:
    def __init__(self, nama, umur):
        self.nama = nama
        self.umur = umur
    
    def info(self):
        return f"{self.nama}, {self.umur} tahun"

# Child class (subclass)
class Mahasiswa(Orang):
    def __init__(self, nama, umur, nim):
        super().__init__(nama, umur)  # Panggil __init__ parent
        self.nim = nim
    
    def info(self):
        return f"{super().info()}, NIM: {self.nim}"

class Dosen(Orang):
    def __init__(self, nama, umur, nip):
        super().__init__(nama, umur)
        self.nip = nip
    
    def info(self):
        return f"{super().info()}, NIP: {self.nip}"

mhs = Mahasiswa("Ahmad", 20, "12345")
dosen = Dosen("Dr. Budi", 45, "98765")

print(mhs.info())    # Ahmad, 20 tahun, NIM: 12345
print(dosen.info())  # Dr. Budi, 45 tahun, NIP: 98765
```

### Mengecek Inheritance

```python
print(isinstance(mhs, Mahasiswa))  # True
print(isinstance(mhs, Orang))      # True
print(issubclass(Mahasiswa, Orang))  # True
```

## Encapsulation

Menyembunyikan detail implementasi:

```python
class BankAccount:
    def __init__(self, saldo_awal):
        self._saldo = saldo_awal  # Protected (konvensi)
    
    def deposit(self, jumlah):
        if jumlah > 0:
            self._saldo += jumlah
            return True
        return False
    
    def withdraw(self, jumlah):
        if 0 < jumlah <= self._saldo:
            self._saldo -= jumlah
            return True
        return False
    
    def get_saldo(self):
        return self._saldo

akun = BankAccount(1000000)
akun.deposit(500000)
akun.withdraw(200000)
print(akun.get_saldo())  # 1300000
```

### Private Attributes

```python
class Rahasia:
    def __init__(self):
        self.__private = "Ini private"  # Name mangling
    
    def get_private(self):
        return self.__private

obj = Rahasia()
# print(obj.__private)  # Error: AttributeError
print(obj.get_private())  # Ini private
print(obj._Rahasia__private)  # Bisa diakses tapi tidak disarankan
```

## Property Decorator

Mengakses method seperti atribut:

```python
class Lingkaran:
    def __init__(self, radius):
        self._radius = radius
    
    @property
    def radius(self):
        return self._radius
    
    @radius.setter
    def radius(self, value):
        if value < 0:
            raise ValueError("Radius tidak boleh negatif")
        self._radius = value
    
    @property
    def luas(self):
        return 3.14159 * self._radius ** 2
    
    @property
    def keliling(self):
        return 2 * 3.14159 * self._radius

lingkaran = Lingkaran(5)
print(lingkaran.radius)    # 5
print(lingkaran.luas)      # 78.53975
lingkaran.radius = 10
print(lingkaran.luas)      # 314.159
```

## Polymorphism

Objek berbeda merespons method yang sama dengan cara berbeda:

```python
class Hewan:
    def suara(self):
        pass

class Kucing(Hewan):
    def suara(self):
        return "Meow!"

class Anjing(Hewan):
    def suara(self):
        return "Woof!"

class Sapi(Hewan):
    def suara(self):
        return "Moo!"

def buat_suara(hewan):
    print(hewan.suara())

hewan_list = [Kucing(), Anjing(), Sapi()]
for hewan in hewan_list:
    buat_suara(hewan)
# Output:
# Meow!
# Woof!
# Moo!
```

## Class Method dan Static Method

```python
class Mahasiswa:
    jumlah = 0
    
    def __init__(self, nama):
        self.nama = nama
        Mahasiswa.jumlah += 1
    
    @classmethod
    def get_jumlah(cls):
        """Method yang bekerja dengan class, bukan instance"""
        return cls.jumlah
    
    @classmethod
    def from_string(cls, data_string):
        """Factory method untuk membuat instance dari string"""
        nama = data_string.split("-")[0]
        return cls(nama)
    
    @staticmethod
    def validasi_nim(nim):
        """Method yang tidak memerlukan instance atau class"""
        return len(nim) == 5 and nim.isdigit()

# Penggunaan
mhs1 = Mahasiswa("Ahmad")
mhs2 = Mahasiswa.from_string("Budi-12345")

print(Mahasiswa.get_jumlah())  # 2
print(Mahasiswa.validasi_nim("12345"))  # True
print(Mahasiswa.validasi_nim("abc"))    # False
```

## Dataclass (Python 3.7+)

Cara ringkas membuat class untuk menyimpan data:

```python
from dataclasses import dataclass, field

@dataclass
class Mahasiswa:
    nama: str
    nim: str
    jurusan: str = "Informatika"
    nilai: list = field(default_factory=list)
    
    def rata_rata(self):
        if not self.nilai:
            return 0
        return sum(self.nilai) / len(self.nilai)

mhs = Mahasiswa("Ahmad", "12345")
print(mhs)  # Mahasiswa(nama='Ahmad', nim='12345', jurusan='Informatika', nilai=[])

mhs.nilai = [85, 90, 88]
print(mhs.rata_rata())  # 87.666...

# Otomatis mengimplementasikan __eq__
mhs2 = Mahasiswa("Ahmad", "12345")
print(mhs == mhs2)  # True (jika nilai sama)
```

## Abstract Base Class

Mendefinisikan interface yang harus diimplementasikan:

```python
from abc import ABC, abstractmethod

class Bentuk(ABC):
    @abstractmethod
    def luas(self):
        pass
    
    @abstractmethod
    def keliling(self):
        pass

class Persegi(Bentuk):
    def __init__(self, sisi):
        self.sisi = sisi
    
    def luas(self):
        return self.sisi ** 2
    
    def keliling(self):
        return 4 * self.sisi

class Lingkaran(Bentuk):
    def __init__(self, radius):
        self.radius = radius
    
    def luas(self):
        return 3.14159 * self.radius ** 2
    
    def keliling(self):
        return 2 * 3.14159 * self.radius

# bentuk = Bentuk()  # Error: tidak bisa instantiate ABC
persegi = Persegi(5)
print(persegi.luas())  # 25
```

## Composition

Menggunakan objek lain sebagai komponen:

```python
class Mesin:
    def __init__(self, tenaga):
        self.tenaga = tenaga
    
    def nyalakan(self):
        return f"Mesin {self.tenaga} HP menyala"

class Roda:
    def __init__(self, ukuran):
        self.ukuran = ukuran

class Mobil:
    def __init__(self, merk, tenaga_mesin):
        self.merk = merk
        self.mesin = Mesin(tenaga_mesin)  # Composition
        self.roda = [Roda(17) for _ in range(4)]
    
    def start(self):
        return f"{self.merk}: {self.mesin.nyalakan()}"

mobil = Mobil("Toyota", 150)
print(mobil.start())  # Toyota: Mesin 150 HP menyala
```

## Ringkasan Konsep OOP

| Konsep | Deskripsi |
|--------|-----------|
| Class | Blueprint untuk membuat objek |
| Object | Instance dari class |
| Attribute | Data yang disimpan dalam objek |
| Method | Fungsi yang didefinisikan dalam class |
| Inheritance | Mewarisi atribut/method dari class lain |
| Encapsulation | Menyembunyikan detail implementasi |
| Polymorphism | Objek berbeda merespons method yang sama |
| Abstraction | Menyederhanakan kompleksitas |

## Latihan

1. Buat class `Buku` dengan atribut judul, penulis, tahun, dan method untuk menampilkan info
2. Buat class `Perpustakaan` yang menyimpan koleksi buku dengan method pinjam dan kembalikan
3. Buat hierarki class untuk bangun datar (Persegi, Persegi Panjang, Segitiga, Lingkaran)
4. Implementasikan class `Playlist` yang menyimpan lagu dengan operasi add, remove, shuffle
