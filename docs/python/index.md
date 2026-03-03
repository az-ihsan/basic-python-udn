# Dasar Python

Python adalah bahasa pemrograman tingkat tinggi yang mudah dipelajari dan sangat populer di bidang ilmu data dan kecerdasan buatan. Bagian ini membahas dasar-dasar pemrograman Python.

## Mengapa Python?

- **Mudah dipelajari** - Sintaks yang bersih dan mudah dibaca
- **Serbaguna** - Dapat digunakan untuk berbagai keperluan
- **Ekosistem yang kaya** - Banyak pustaka untuk komputasi ilmiah
- **Komunitas besar** - Banyak sumber belajar dan dukungan

## Menjalankan Python

### Mode Interaktif

Ketik `python` atau `python3` di terminal:

```python
>>> print("Halo, Dunia!")
Halo, Dunia!
>>> 2 + 3
5
```

### Menjalankan Berkas Skrip

Simpan kode dalam file `.py` dan jalankan:

```bash
python script.py
```

## Daftar Materi

```{toctree}
:maxdepth: 1

tipe_data
kontrol_alur
fungsi
itertools_combinatorik
modul_dan_paket
file_io
error_handling
oop_ringkas
```

## Hello World

Program pertama dalam Python:

```python
print("Halo, Dunia!")
```

Output:
```
Halo, Dunia!
```

```{doctest}
>>> print("Halo, Dunia!")
Halo, Dunia!
```

## Komentar dan Docstring

Komentar digunakan untuk menjelaskan kode dan tidak dieksekusi. Dalam Python, komentar ditulis dengan `#`.

Docstring berbeda dari komentar. Docstring adalah string literal yang menjadi dokumentasi objek (modul, fungsi, kelas) jika diletakkan sebagai pernyataan pertama.

```python
# Ini adalah komentar satu baris
# Ini adalah komentar multi-baris
# yang ditulis dengan beberapa tanda pagar.

x = 5  # Komentar di akhir baris

def luas_lingkaran(r):
    """Menghitung luas lingkaran berdasarkan jari-jari."""
    return 3.14159 * r ** 2
```

## Variabel

Variabel menyimpan nilai dalam memori:

```python
# Tidak perlu deklarasi tipe
nama = "Ahmad"
umur = 25
tinggi = 175.5
aktif = True

print(nama)   # Ahmad
print(umur)   # 25
```

### Aturan Penamaan Variabel

- Dimulai dengan huruf atau underscore (`_`)
- Tidak boleh dimulai dengan angka
- Hanya boleh mengandung huruf, angka, dan underscore
- Peka huruf besar/kecil (`nama` dan `Nama` berbeda)

```python
# Valid
nama_mahasiswa = "Budi"
_private = 10
nomor1 = 100

# Tidak valid
# 1angka = 5      # Dimulai dengan angka
# nama-user = ""  # Mengandung tanda minus
```

## Langkah Selanjutnya

Lanjutkan ke [Tipe Data](tipe_data.md) untuk mempelajari berbagai tipe data dalam Python.
