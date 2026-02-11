# spaCy

[spaCy](https://spacy.io/) adalah pustaka NLP modern yang dirancang untuk penggunaan produksi. spaCy menyediakan pipeline lengkap untuk pemrosesan teks dengan performa tinggi.

## Mengapa spaCy?

- **Cepat** - Dioptimasi untuk performa tinggi
- **Akurat** - Model terlatih dengan akurasi tinggi
- **Mudah digunakan** - API yang intuitif dan konsisten
- **Lengkap** - Tokenisasi, POS tagging, NER, lemmatization, dan banyak lagi
- **Multibahasa** - Mendukung banyak bahasa termasuk Indonesia dan Inggris

## Instalasi

```bash
pip install spacy
```

Atau jika menggunakan uv:

```bash
uv add spacy
```

### Mengunduh Model Bahasa

spaCy memerlukan model bahasa untuk bekerja. Berikut cara mengunduh model:

**Model Bahasa Inggris:**

```bash
python -m spacy download en_core_web_sm
```

**Model Bahasa Indonesia:**

```bash
python -m spacy download id_core_news_sm
```

:::{note}
Model `_sm` (small) cocok untuk pembelajaran. Untuk produksi, pertimbangkan model `_md` (medium) atau `_lg` (large) yang lebih akurat.
:::

## Daftar Materi

```{toctree}
:maxdepth: 1

dasar
preprocessing
tokenisasi_pos_ner
```

## Contoh Cepat

```python
import spacy

# Load model bahasa Inggris
nlp = spacy.load("en_core_web_sm")

# Proses teks
doc = nlp("Apple is looking at buying U.K. startup for $1 billion")

# Lihat token dan atributnya
for token in doc:
    print(f"{token.text:12} {token.pos_:6} {token.dep_:10}")
```

Output:

```
Apple        PROPN  nsubj     
is           AUX    aux       
looking      VERB   ROOT      
at           ADP    prep      
buying       VERB   pcomp     
U.K.         PROPN  dobj      
startup      NOUN   dobj      
for          ADP    prep      
$            SYM    quantmod  
1            NUM    compound  
billion      NUM    pobj      
```

## Contoh dengan Bahasa Indonesia

```python
import spacy

# Load model bahasa Indonesia
nlp = spacy.load("id_core_news_sm")

# Proses teks
doc = nlp("Jakarta adalah ibu kota Indonesia yang terletak di Pulau Jawa.")

# Lihat token
for token in doc:
    print(f"{token.text:15} {token.pos_:6} {token.lemma_}")
```

Output:

```
Jakarta         PROPN  Jakarta
adalah          AUX    adalah
ibu             NOUN   ibu
kota            NOUN   kota
Indonesia       PROPN  Indonesia
yang            PRON   yang
terletak        VERB   letak
di              ADP    di
Pulau           PROPN  Pulau
Jawa            PROPN  Jawa
.               PUNCT  .
```

## Pipeline spaCy

Ketika Anda memproses teks dengan spaCy, teks tersebut melewati serangkaian komponen pipeline:

```
┌────────────┐    ┌───────────┐    ┌────────┐    ┌───────┐
│  Tokenizer │ → │   Tagger  │ → │ Parser │ → │  NER  │
└────────────┘    └───────────┘    └────────┘    └───────┘
      ↓                ↓               ↓            ↓
   Token           POS Tags      Dependencies   Entities
```

Setiap komponen menambahkan anotasi ke dokumen yang dapat Anda akses.

## Langkah Selanjutnya

Lanjutkan ke [spaCy Dasar](dasar.md) untuk mempelajari konsep dasar dan objek-objek utama dalam spaCy.
