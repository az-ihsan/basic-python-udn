# Dasar Python

Python adalah bahasa pemrograman tingkat tinggi yang mudah dipelajari dan sangat populer di bidang Data Science dan Kecerdasan Buatan. Bagian ini akan membahas dasar-dasar pemrograman Python.

## Mengapa Python?

- **Mudah dipelajari** - Sintaks yang bersih dan mudah dibaca
- **Versatile** - Dapat digunakan untuk berbagai keperluan
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

### Menjalankan File Script

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

## Komentar

Komentar digunakan untuk menjelaskan kode dan tidak dieksekusi:

```python
# Ini adalah komentar satu baris

"""
Ini adalah komentar
multi-baris (docstring)
"""

x = 5  # Komentar di akhir baris
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
- Case-sensitive (`nama` dan `Nama` berbeda)

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
