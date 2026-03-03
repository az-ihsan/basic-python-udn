# Itertools dan Kombinatorik untuk Probabilitas & Statistik

Modul `math` dan `itertools` sangat berguna untuk menghitung **banyaknya cara** dalam masalah probabilitas dan statistik:

- berapa banyak cara menyusun objek (permutasi),
- berapa banyak cara memilih objek (kombinasi),
- berapa besar ruang sampel percobaan (Cartesian product),
- serta menghitung peluang dari ruang sampel tersebut.

Di sini kita akan fokus pada fungsi-fungsi yang paling relevan untuk kuliah Probabilitas & Statistik.

---

## Faktorial

Secara matematis, **faktorial** didefinisikan sebagai:

\[
n! = 1 \times 2 \times 3 \times \dots \times n,\quad 0! = 1
\]

Di Python, faktorial tersedia di modul `math`:

```python
import math

print(math.factorial(5))   # 120  (5! = 1*2*3*4*5)
print(math.factorial(0))   # 1    (0! didefinisikan = 1)

# Contoh: berapa banyak cara menyusun 4 orang dalam 4 kursi?
n = 4
print(math.factorial(n))   # 24 cara
```

Faktorial adalah blok bangunan utama untuk permutasi dan kombinasi.

---

## Permutasi (Urutan Penting)

Dalam probabilitas, **permutasi** menghitung banyaknya cara menyusun \(r\) objek dari \(n\) objek **dengan memperhatikan urutan**:

\[
P(n, r) = \frac{n!}{(n - r)!}
\]

Di Python, kita bisa:

- menggunakan rumus dengan `math.factorial`, atau
- menghasilkan semua susunan dengan `itertools.permutations`.

### Rumus dengan `math.factorial`

```python
import math

def permutasi_rumus(n, r):
    return math.factorial(n) // math.factorial(n - r)

print(permutasi_rumus(5, 2))  # 20
print(permutasi_rumus(4, 4))  # 24
```

### `itertools.permutations`

```python
from itertools import permutations

items = ["A", "B", "C"]

print("Permutasi 2 dari [A, B, C]:")
for p in permutations(items, 2):
    print(p)

# Hitung banyaknya permutasi 2 dari 3
all_perm = list(permutations(items, 2))
print("Jumlah permutasi:", len(all_perm))  # 6
```

Ini sesuai dengan rumus:

\[
P(3, 2) = \frac{3!}{(3-2)!} = \frac{3!}{1!} = 6
\]

---

## Kombinasi (Urutan Tidak Penting)

**Kombinasi** menghitung banyaknya cara memilih \(r\) objek dari \(n\) objek **tanpa memperhatikan urutan**:

\[
C(n, r) = \binom{n}{r} = \frac{n!}{r!(n - r)!}
\]

### Rumus dengan `math.factorial`

```python
import math

def kombinasi_rumus(n, r):
    return math.factorial(n) // (math.factorial(r) * math.factorial(n - r))

print(kombinasi_rumus(5, 2))  # 10
print(kombinasi_rumus(6, 3))  # 20
```

### `itertools.combinations`

```python
from itertools import combinations

items = ["A", "B", "C"]

print("Kombinasi 2 dari [A, B, C]:")
for c in combinations(items, 2):
    print(c)

all_comb = list(combinations(items, 2))
print("Jumlah kombinasi:", len(all_comb))  # 3
```

Di sini hanya ada 3 kombinasi:

- `("A", "B")`
- `("A", "C")`
- `("B", "C")`

Sedangkan susunan `("B", "A")` tidak dihitung terpisah karena urutan tidak penting.

---

## Kombinasi dengan Pengulangan

Kadang kita boleh **mengambil elemen yang sama lebih dari sekali**. Secara matematis, banyaknya kombinasi dengan pengulangan dari \(n\) jenis objek yang diambil \(r\) kali adalah:

\[
\binom{n + r - 1}{r}
\]

Di Python, kita bisa menggunakan `itertools.combinations_with_replacement`:

```python
from itertools import combinations_with_replacement

rasa = ["vanila", "coklat", "stroberi"]

print("Pilih 2 scoop es krim (boleh rasa yang sama):")
for combo in combinations_with_replacement(rasa, 2):
    print(combo)
```

Contoh di atas memodelkan situasi seperti “stars and bars” di kombinatorik (membagi `r` item identik ke `n` kategori).

---

## Cartesian Product dan Ruang Sampel

Dalam probabilitas, **ruang sampel** untuk dua percobaan independen dapat dibentuk dengan **Cartesian product**:

\[
S = A \times B = \{(a, b) \mid a \in A,\; b \in B\}
\]

Di Python, ini dilakukan dengan `itertools.product`.

### Contoh: Dua kali lempar koin

```python
from itertools import product

koin = ["H", "T"]  # H = Head, T = Tail

ruang_sampel = list(product(koin, repeat=2))

print("Ruang sampel dua kali lempar koin:")
for outcome in ruang_sampel:
    print(outcome)

print("Jumlah outcome:", len(ruang_sampel))  # 4
```

Hasil:

- `("H", "H")`
- `("H", "T")`
- `("T", "H")`
- `("T", "T")`

Sehingga peluang mendapatkan tepat satu kali `H` adalah:

\[
P(\text{satu H}) = \frac{2}{4} = \frac{1}{2}
\]

### Contoh: Dua dadu enam sisi

```python
from itertools import product

dadu = [1, 2, 3, 4, 5, 6]

ruang_sampel = list(product(dadu, repeat=2))
print("Jumlah outcome:", len(ruang_sampel))  # 36

# Peluang jumlah mata dadu = 7
jumlah_7 = [pair for pair in ruang_sampel if sum(pair) == 7]

print("Outcome dengan jumlah 7:", jumlah_7)
print("Banyaknya:", len(jumlah_7))
print("Peluang:", len(jumlah_7) / len(ruang_sampel))
```

Secara teori, memang ada 6 pasangan yang jumlahnya 7:

- (1, 6), (2, 5), (3, 4), (4, 3), (5, 2), (6, 1)

Jadi:

\[
P(\text{jumlah} = 7) = \frac{6}{36} = \frac{1}{6}
\]

---

## Contoh Aplikasi Terpadu

### Contoh 1: Memilih Panitia (Kombinasi)

Misalkan ada 10 mahasiswa dan akan dipilih 3 orang sebagai panitia (tanpa peran khusus). Berapa banyak susunan panitia yang mungkin?

```python
import math

n = 10  # jumlah mahasiswa
r = 3   # jumlah yang dipilih

jumlah_cara = math.factorial(n) // (math.factorial(r) * math.factorial(n - r))
print("Banyak cara memilih panitia:", jumlah_cara)  # 120
```

Jika setiap panitia dipilih secara acak dari 10 mahasiswa tanpa penggantian, maka:

\[
P(\text{suatu kombinasi tertentu}) = \frac{1}{\binom{10}{3}} = \frac{1}{120}
\]

### Contoh 2: Permutasi Kursi (Urutan Penting)

Ada 5 mahasiswa dan 3 kursi di baris depan. Berapa banyak cara menyusun siapa duduk di mana?

```python
import math

n = 5
r = 3

jumlah_cara = math.factorial(n) // math.factorial(n - r)
print("Banyak cara menyusun mahasiswa di kursi:", jumlah_cara)  # 60
```

Di sini urutan kursi penting, sehingga kita memakai permutasi, bukan kombinasi.

---

## Ringkasan

- Gunakan **`math.factorial`** untuk menghitung \(n!\) yang dipakai dalam rumus permutasi dan kombinasi.
- Gunakan **`itertools.permutations`** ketika urutan penting (penyusunan).
- Gunakan **`itertools.combinations`** ketika urutan tidak penting (pemilihan).
- Gunakan **`itertools.combinations_with_replacement`** ketika pemilihan boleh mengulang elemen yang sama.
- Gunakan **`itertools.product`** untuk membangun **ruang sampel** percobaan majemuk (Cartesian product), sehingga peluang bisa dihitung sebagai \(\text{jumlah kasus menguntungkan} / \text{jumlah semua kasus}\).

