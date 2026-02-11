# Kontrol Alur

Kontrol alur memungkinkan program untuk membuat keputusan dan mengulang tindakan. Mari pelajari struktur kontrol utama dalam Python.

## Percabangan (Conditional)

### if Statement

```python
umur = 18

if umur >= 17:
    print("Anda boleh memiliki SIM")
```

### if-else Statement

```python
nilai = 75

if nilai >= 60:
    print("Lulus")
else:
    print("Tidak Lulus")
```

### if-elif-else Statement

```python
nilai = 85

if nilai >= 90:
    grade = "A"
elif nilai >= 80:
    grade = "B"
elif nilai >= 70:
    grade = "C"
elif nilai >= 60:
    grade = "D"
else:
    grade = "E"

print(f"Grade Anda: {grade}")  # Grade Anda: B
```

### Nested if

```python
umur = 20
punya_sim = True

if umur >= 17:
    if punya_sim:
        print("Boleh mengemudi")
    else:
        print("Belum punya SIM")
else:
    print("Belum cukup umur")
```

### Conditional Expression (Ternary Operator)

```python
umur = 20
status = "Dewasa" if umur >= 18 else "Anak-anak"
print(status)  # Dewasa
```

## Perulangan (Loop)

### for Loop

Digunakan untuk mengiterasi sequence (list, tuple, string, range, dll):

```python
# Iterasi list
buah = ["apel", "jeruk", "mangga"]
for b in buah:
    print(b)

# Output:
# apel
# jeruk
# mangga
```

### range()

Fungsi untuk menghasilkan urutan angka:

```python
# range(stop)
for i in range(5):
    print(i, end=" ")  # 0 1 2 3 4

print()

# range(start, stop)
for i in range(2, 6):
    print(i, end=" ")  # 2 3 4 5

print()

# range(start, stop, step)
for i in range(0, 10, 2):
    print(i, end=" ")  # 0 2 4 6 8
```

### enumerate()

Mendapatkan indeks dan nilai sekaligus:

```python
buah = ["apel", "jeruk", "mangga"]

for i, b in enumerate(buah):
    print(f"{i}: {b}")

# Output:
# 0: apel
# 1: jeruk
# 2: mangga
```

### zip()

Menggabungkan beberapa iterables:

```python
nama = ["Ahmad", "Budi", "Citra"]
nilai = [85, 90, 78]

for n, v in zip(nama, nilai):
    print(f"{n}: {v}")

# Output:
# Ahmad: 85
# Budi: 90
# Citra: 78
```

### while Loop

Mengulang selama kondisi bernilai True:

```python
counter = 0

while counter < 5:
    print(counter)
    counter += 1

# Output:
# 0
# 1
# 2
# 3
# 4
```

### Infinite Loop

Hati-hati dengan loop tak terbatas:

```python
# Contoh infinite loop (jangan dijalankan tanpa cara keluar)
# while True:
#     print("Loop selamanya!")

# Infinite loop dengan kondisi keluar
while True:
    jawaban = input("Ketik 'keluar' untuk berhenti: ")
    if jawaban == "keluar":
        break
```

## Kontrol Loop

### break

Menghentikan loop sepenuhnya:

```python
for i in range(10):
    if i == 5:
        break
    print(i)

# Output:
# 0
# 1
# 2
# 3
# 4
```

### continue

Melewati iterasi saat ini dan lanjut ke iterasi berikutnya:

```python
for i in range(10):
    if i % 2 == 0:
        continue
    print(i)

# Output (hanya bilangan ganjil):
# 1
# 3
# 5
# 7
# 9
```

### else pada Loop

Blok else dieksekusi jika loop selesai tanpa break:

```python
# Mencari bilangan prima
n = 17

for i in range(2, n):
    if n % i == 0:
        print(f"{n} bukan bilangan prima")
        break
else:
    print(f"{n} adalah bilangan prima")

# Output: 17 adalah bilangan prima
```

## Nested Loop

Loop di dalam loop:

```python
# Tabel perkalian
for i in range(1, 4):
    for j in range(1, 4):
        print(f"{i} x {j} = {i * j}")
    print()  # Baris kosong

# Output:
# 1 x 1 = 1
# 1 x 2 = 2
# 1 x 3 = 3
#
# 2 x 1 = 2
# 2 x 2 = 4
# 2 x 3 = 6
#
# 3 x 1 = 3
# 3 x 2 = 6
# 3 x 3 = 9
```

## Match-Case (Python 3.10+)

Pattern matching yang mirip switch-case:

```python
status_code = 404

match status_code:
    case 200:
        print("OK")
    case 404:
        print("Not Found")
    case 500:
        print("Internal Server Error")
    case _:
        print("Unknown Status")

# Output: Not Found
```

### Pattern Matching dengan Struktur Data

```python
point = (0, 5)

match point:
    case (0, 0):
        print("Origin")
    case (0, y):
        print(f"Pada sumbu Y, y={y}")
    case (x, 0):
        print(f"Pada sumbu X, x={x}")
    case (x, y):
        print(f"Titik ({x}, {y})")

# Output: Pada sumbu Y, y=5
```

## Contoh Aplikasi

### Menghitung Faktorial

```python
n = 5
faktorial = 1

for i in range(1, n + 1):
    faktorial *= i

print(f"{n}! = {faktorial}")  # 5! = 120
```

### Menampilkan Deret Fibonacci

```python
n = 10
a, b = 0, 1

print("Deret Fibonacci:")
for _ in range(n):
    print(a, end=" ")
    a, b = b, a + b

# Output: 0 1 1 2 3 5 8 13 21 34
```

### Validasi Input

```python
while True:
    try:
        umur = int(input("Masukkan umur Anda: "))
        if umur < 0:
            print("Umur tidak boleh negatif!")
            continue
        break
    except ValueError:
        print("Masukkan angka yang valid!")

print(f"Umur Anda: {umur}")
```

## Latihan

1. Buat program untuk menentukan apakah suatu bilangan ganjil atau genap
2. Buat program untuk mencetak bilangan 1-100 yang habis dibagi 3 dan 5
3. Buat program untuk mencari bilangan prima dari 1 sampai 50
4. Buat program tebak angka (komputer memilih angka random, user menebak)
