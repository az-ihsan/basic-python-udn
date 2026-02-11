# Setup Environment

Sebelum mulai belajar Python, kita perlu menyiapkan lingkungan pengembangan. Bagian ini akan memandu Anda untuk menginstal Python dan alat-alat pendukung lainnya.

## Instalasi Python

### Windows

1. Kunjungi [python.org/downloads](https://www.python.org/downloads/)
2. Unduh installer Python versi terbaru (3.11 atau lebih baru)
3. Jalankan installer dan **centang "Add Python to PATH"**
4. Klik "Install Now"

### macOS

Gunakan Homebrew (disarankan):

```bash
# Instal Homebrew jika belum ada
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Instal Python
brew install python
```

### Linux (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv
```

## Verifikasi Instalasi

Buka terminal atau command prompt dan jalankan:

```bash
python --version
# atau
python3 --version
```

Anda seharusnya melihat output seperti `Python 3.11.x` atau versi yang lebih baru.

## Virtual Environment

Virtual environment adalah lingkungan Python yang terisolasi. Ini penting untuk menghindari konflik antar proyek.

### Membuat Virtual Environment

```bash
# Buat folder proyek
mkdir proyek-python
cd proyek-python

# Buat virtual environment
python -m venv venv

# Aktivasi virtual environment
# Windows:
venv\Scripts\activate

# macOS/Linux:
source venv/bin/activate
```

### Tanda Virtual Environment Aktif

Ketika virtual environment aktif, Anda akan melihat nama environment di awal prompt:

```bash
(venv) $ 
```

### Menonaktifkan Virtual Environment

```bash
deactivate
```

## Menggunakan uv (Opsional)

[uv](https://github.com/astral-sh/uv) adalah package manager Python yang sangat cepat. Untuk menginstalnya:

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Kemudian buat proyek baru dengan:

```bash
uv init proyek-baru
cd proyek-baru
uv sync
```

## Instalasi Pustaka yang Dibutuhkan

Untuk mengikuti materi ini, instal pustaka-pustaka berikut:

```bash
pip install numpy pandas scikit-learn torch matplotlib jupyter spacy transformers
```

Atau jika menggunakan uv:

```bash
uv add numpy pandas scikit-learn torch matplotlib jupyter spacy transformers
```

### Model spaCy

Untuk menggunakan spaCy, Anda perlu mengunduh model bahasa:

```bash
# Model Bahasa Inggris
python -m spacy download en_core_web_sm

# Model Bahasa Indonesia
python -m spacy download id_core_news_sm
```

### Model Transformers

Model Hugging Face Transformers akan diunduh secara otomatis saat pertama kali digunakan dan disimpan di cache lokal (`~/.cache/huggingface/`). Tidak perlu mengunduh model secara manual.

## Editor/IDE yang Disarankan

### Visual Studio Code

- Gratis dan ringan
- Extension Python yang sangat baik
- Unduh di [code.visualstudio.com](https://code.visualstudio.com/)

Setelah instal VS Code, pasang extension:
- Python (Microsoft)
- Pylance
- Jupyter

### Jupyter Notebook

Jupyter sangat berguna untuk eksperimen dan pembelajaran interaktif:

```bash
# Jalankan Jupyter
jupyter notebook
```

Browser akan terbuka dengan antarmuka Jupyter.

## Struktur Folder Proyek

Berikut adalah struktur folder yang disarankan untuk proyek Python:

```
proyek-python/
├── venv/               # Virtual environment
├── src/                # Kode sumber
│   └── __init__.py
├── tests/              # Unit tests
├── notebooks/          # Jupyter notebooks
├── data/               # Data files
├── requirements.txt    # Daftar dependensi
└── README.md           # Dokumentasi proyek
```

## Langkah Selanjutnya

Setelah lingkungan siap, lanjutkan ke bagian [Dasar Python](../python/index.md) untuk mulai belajar pemrograman Python.
