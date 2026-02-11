# Hugging Face Transformers

[Hugging Face Transformers](https://huggingface.co/transformers/) adalah pustaka yang menyediakan akses mudah ke ribuan model pre-trained untuk Natural Language Processing. Dengan Transformers, Anda dapat melakukan berbagai tugas NLP hanya dengan beberapa baris kode.

## Mengapa Transformers?

- **State-of-the-art** - Model terbaik untuk berbagai tugas NLP
- **Pre-trained** - Model sudah dilatih pada data besar, tinggal pakai
- **Mudah digunakan** - API `pipeline` yang sangat sederhana
- **Model Hub** - Ribuan model dari komunitas global
- **Multibahasa** - Banyak model mendukung berbagai bahasa

## Instalasi

```bash
pip install transformers
```

Atau jika menggunakan uv:

```bash
uv add transformers
```

:::{note}
Model akan diunduh secara otomatis saat pertama kali digunakan dan disimpan di cache lokal (`~/.cache/huggingface/`).
:::

## Daftar Materi

```{toctree}
:maxdepth: 1

dasar
```

## Contoh Cepat

```python
from transformers import pipeline

# Buat pipeline untuk analisis sentimen
classifier = pipeline("sentiment-analysis")

# Gunakan pipeline
result = classifier("I love learning about AI!")
print(result)
# [{'label': 'POSITIVE', 'score': 0.9998}]
```

## Contoh NER

```python
from transformers import pipeline

ner = pipeline("ner", aggregation_strategy="simple")

text = "Apple Inc. was founded by Steve Jobs in California."
entities = ner(text)

for entity in entities:
    print(f"{entity['word']:20} → {entity['entity_group']}")
```

Output:

```
Apple Inc.           → ORG
Steve Jobs           → PER
California           → LOC
```

## Perbandingan spaCy vs Transformers

| Aspek | spaCy | Transformers |
|-------|-------|--------------|
| Kecepatan | Sangat cepat | Lebih lambat |
| Akurasi | Baik | State-of-the-art |
| Ukuran model | Kecil (MB) | Besar (100MB-GB) |
| GPU | Opsional | Disarankan |
| Kasus penggunaan | Produksi, real-time | Akurasi maksimal |

## Langkah Selanjutnya

Lanjutkan ke [Transformers Dasar](dasar.md) untuk mempelajari konsep dasar dan berbagai pipeline yang tersedia.
