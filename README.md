# Basic Python UDN

**Sebagai bahan pembantu pembelajaran Kecerdasan Buatan Universitas Darunnajah (UDN)**

📖 **Dokumentasi (online):** [https://az-ihsan.github.io/basic-python-udn/](https://az-ihsan.github.io/basic-python-udn/)

Repositori ini berisi materi pembelajaran Basic Python UDN, mencakup:

- Dasar-dasar Python
- NumPy untuk komputasi numerik
- Pandas untuk analisis data
- Scikit-learn untuk machine learning
- PyTorch untuk deep learning

## Struktur Repositori

```
basic-python-udn/
├── docs/               # Dokumentasi Sphinx
│   ├── conf.py         # Konfigurasi Sphinx
│   ├── index.md        # Halaman utama
│   ├── pengantar/      # Modul pengantar
│   ├── setup/          # Modul instalasi
│   ├── python/         # Modul dasar Python
│   ├── numpy/          # Modul NumPy
│   ├── pandas/         # Modul Pandas
│   ├── scikit_learn/   # Modul scikit-learn
│   └── pytorch/        # Modul PyTorch
├── pyproject.toml      # Konfigurasi proyek
└── README.md           # File ini
```

## Instalasi

### Prasyarat

- Python 3.10 atau lebih baru
- pip atau uv

### Menggunakan pip

```bash
# Clone repositori
git clone <repository-url>
cd basic-python-udn

# Buat virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# atau
venv\Scripts\activate     # Windows

# Instal dependencies
pip install -e .
pip install -e ".[docs]"
```

### Menggunakan uv

```bash
# Clone repositori
git clone <repository-url>
cd basic-python-udn

# Sync dependencies
uv sync

# Aktifkan environment
source .venv/bin/activate  # Linux/macOS
# atau
.venv\Scripts\activate     # Windows
```

## Build Dokumentasi

### Menggunakan sphinx-build

```bash
# Pastikan dependencies docs terinstal
pip install sphinx myst-parser sphinx-rtd-theme

# Build HTML
sphinx-build -b html docs docs/_build/html

# Atau dari folder docs
cd docs
make html
```

### Lihat Dokumentasi

Setelah build, buka `docs/_build/html/index.html` di browser.

```bash
# macOS
open docs/_build/html/index.html

# Linux
xdg-open docs/_build/html/index.html

# Windows
start docs/_build/html/index.html
```

### Live Reload (Opsional)

Untuk development dengan auto-reload:

```bash
pip install sphinx-autobuild
sphinx-autobuild docs docs/_build/html
```

Kemudian buka http://127.0.0.1:8000 di browser.

### Deploy ke GitHub Pages

Dokumentasi dapat di-deploy otomatis ke GitHub Pages dengan GitHub Actions:

1. **Aktifkan GitHub Pages dari Actions**  
   Di repositori: **Settings → Pages → Build and deployment → Source** pilih **GitHub Actions**.

2. **Push ke `main`**  
   Workflow `.github/workflows/deploy-docs.yml` akan membangun Sphinx dan mendeploy ke `https://<username>.github.io/<repo>/` (atau custom domain jika dikonfigurasi).

3. **Jalankan manual (opsional)**  
   **Actions → Deploy docs to GitHub Pages → Run workflow**.

## Penggunaan

### Jupyter Notebook

```bash
# Jalankan Jupyter
jupyter notebook

# Atau Jupyter Lab
jupyter lab
```

### Menjalankan Contoh Kode

```python
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

# Mulai eksperimen!
```

## Kontribusi

Materi ini dikembangkan oleh **Dr. -Ing. Ahmad Z. Ihsan** untuk mahasiswa Universitas Darunnajah.

Jika Anda menemukan kesalahan atau ingin berkontribusi, silakan buat issue atau pull request.

## Lisensi

MIT License 
