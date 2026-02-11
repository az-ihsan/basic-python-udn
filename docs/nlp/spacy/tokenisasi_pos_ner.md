# Tokenisasi, POS Tagging & NER

Bagian ini membahas tiga komponen penting dalam NLP: tokenisasi, Part-of-Speech (POS) tagging, dan Named Entity Recognition (NER).

## Tokenisasi

Tokenisasi adalah proses memecah teks menjadi unit-unit kecil yang disebut token. Token bisa berupa kata, tanda baca, atau simbol.

### Tokenisasi Dasar

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Dr. Smith bought 500 shares of Apple Inc. for $50.")

for token in doc:
    print(f"{token.i:3} | {token.text:10} | {token.is_alpha} | {token.is_punct}")
```

Output:

```
  0 | Dr.        | False | False
  1 | Smith      | True  | False
  2 | bought     | True  | False
  3 | 500        | False | False
  4 | shares     | True  | False
  5 | of         | True  | False
  6 | Apple      | True  | False
  7 | Inc.       | False | False
  8 | for        | True  | False
  9 | $          | False | False
 10 | 50         | False | False
 11 | .          | False | True
```

:::{note}
Perhatikan bahwa "Dr." dan "Inc." tetap utuh sebagai satu token karena spaCy mengenali singkatan umum.
:::

### Tokenisasi Kalimat

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Hello world. This is a test. How are you?")

# Iterasi kalimat
for i, sent in enumerate(doc.sents):
    print(f"Kalimat {i+1}: {sent.text}")
```

Output:

```
Kalimat 1: Hello world.
Kalimat 2: This is a test.
Kalimat 3: How are you?
```

### Tokenisasi Bahasa Indonesia

```python
import spacy

nlp = spacy.load("id_core_news_sm")
doc = nlp("Bpk. Joko membeli 10 buku di Toko Gramedia. Harganya Rp. 150.000.")

for token in doc:
    print(f"{token.text:15} | Index: {token.i}")
```

## Part-of-Speech (POS) Tagging

POS tagging adalah proses menandai setiap token dengan kategori gramatikalnya (kata benda, kata kerja, dll).

### POS Tags di spaCy

spaCy menyediakan dua jenis POS tag:

- `pos_` - Tag kasar (coarse-grained) menggunakan Universal POS tags
- `tag_` - Tag halus (fine-grained) yang lebih spesifik

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("She sells seashells by the seashore.")

print(f"{'Token':<12} {'POS':<8} {'Tag':<8} {'Penjelasan'}")
print("-" * 50)

for token in doc:
    print(f"{token.text:<12} {token.pos_:<8} {token.tag_:<8} {spacy.explain(token.tag_)}")
```

Output:

```
Token        POS      Tag      Penjelasan
--------------------------------------------------
She          PRON     PRP      pronoun, personal
sells        VERB     VBZ      verb, 3rd person singular present
seashells    NOUN     NNS      noun, plural
by           ADP      IN       conjunction, subordinating or preposition
the          DET      DT       determiner
seashore     NOUN     NN       noun, singular or mass
.            PUNCT    .        punctuation mark, sentence closer
```

### Daftar Universal POS Tags

| Tag | Deskripsi | Contoh |
|-----|-----------|--------|
| ADJ | Adjective | big, beautiful |
| ADP | Adposition | in, on, at |
| ADV | Adverb | very, quickly |
| AUX | Auxiliary | is, has, will |
| CONJ | Conjunction | and, but, or |
| DET | Determiner | a, the, this |
| NOUN | Noun | cat, house, idea |
| NUM | Numeral | one, 2, third |
| PRON | Pronoun | I, you, he |
| PROPN | Proper Noun | John, London |
| PUNCT | Punctuation | . , ! |
| VERB | Verb | run, eat, is |

### POS Tagging Bahasa Indonesia

```python
import spacy

nlp = spacy.load("id_core_news_sm")
doc = nlp("Mahasiswa itu sedang membaca buku di perpustakaan.")

for token in doc:
    print(f"{token.text:<15} {token.pos_:<8} {token.tag_}")
```

Output:

```
Mahasiswa       NOUN     NN
itu             DET      DT
sedang          ADV      RB
membaca         VERB     VB
buku            NOUN     NN
di              ADP      IN
perpustakaan    NOUN     NN
.               PUNCT    Z
```

### Menggunakan POS untuk Filtering

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The quick brown fox jumps over the lazy dog.")

# Ekstrak hanya kata benda dan kata kerja
nouns = [token.text for token in doc if token.pos_ == "NOUN"]
verbs = [token.text for token in doc if token.pos_ == "VERB"]

print(f"Nouns: {nouns}")
print(f"Verbs: {verbs}")
```

Output:

```
Nouns: ['fox', 'dog']
Verbs: ['jumps']
```

## Named Entity Recognition (NER)

NER adalah proses mengidentifikasi dan mengklasifikasi entitas bernama dalam teks, seperti nama orang, organisasi, lokasi, dll.

### NER Dasar

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple Inc. was founded by Steve Jobs in Cupertino, California in 1976.")

for ent in doc.ents:
    print(f"{ent.text:<20} {ent.label_:<12} {spacy.explain(ent.label_)}")
```

Output:

```
Apple Inc.           ORG          Companies, agencies, institutions, etc.
Steve Jobs           PERSON       People, including fictional
Cupertino            GPE          Countries, cities, states
California           GPE          Countries, cities, states
1976                 DATE         Absolute or relative dates or periods
```

### Daftar Entity Labels Umum

| Label | Deskripsi |
|-------|-----------|
| PERSON | Nama orang |
| ORG | Organisasi, perusahaan |
| GPE | Negara, kota, wilayah (Geo-Political Entity) |
| LOC | Lokasi non-GPE (gunung, sungai) |
| DATE | Tanggal atau periode waktu |
| TIME | Waktu dalam sehari |
| MONEY | Nilai uang |
| PERCENT | Persentase |
| PRODUCT | Produk (bukan layanan) |
| EVENT | Event bernama (perang, olimpiade) |

### NER Bahasa Indonesia

```python
import spacy

nlp = spacy.load("id_core_news_sm")
doc = nlp("Presiden Joko Widodo mengunjungi Yogyakarta pada hari Senin.")

for ent in doc.ents:
    print(f"{ent.text:<25} {ent.label_}")
```

Output:

```
Joko Widodo               PER
Yogyakarta                LOC
Senin                     DAT
```

### Mengakses Posisi Entitas

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Microsoft was founded in Albuquerque.")

for ent in doc.ents:
    print(f"Entity: {ent.text}")
    print(f"  Label: {ent.label_}")
    print(f"  Start char: {ent.start_char}")
    print(f"  End char: {ent.end_char}")
    print(f"  Start token: {ent.start}")
    print(f"  End token: {ent.end}")
    print()
```

### Visualisasi NER dengan displaCy

spaCy menyediakan visualizer bawaan untuk NER:

```python
import spacy
from spacy import displacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple is looking at buying U.K. startup for $1 billion")

# Render di Jupyter Notebook
displacy.render(doc, style="ent", jupyter=True)

# Atau simpan ke file HTML
html = displacy.render(doc, style="ent")
with open("entities.html", "w") as f:
    f.write(html)
```

### Contoh: Ekstraksi Entitas dari Berita

```python
import spacy

nlp = spacy.load("en_core_web_sm")

news_text = """
Google announced a new AI model yesterday in San Francisco. 
CEO Sundar Pichai said the company invested $10 billion in the project.
The model will be available starting January 2026.
"""

doc = nlp(news_text)

# Kelompokkan entitas berdasarkan label
entities_by_type = {}
for ent in doc.ents:
    if ent.label_ not in entities_by_type:
        entities_by_type[ent.label_] = []
    entities_by_type[ent.label_].append(ent.text)

for label, entities in entities_by_type.items():
    print(f"{label}: {entities}")
```

Output:

```
ORG: ['Google']
DATE: ['yesterday', 'January 2026']
GPE: ['San Francisco']
PERSON: ['Sundar Pichai']
MONEY: ['$10 billion']
```

## Dependency Parsing

Selain POS dan NER, spaCy juga melakukan dependency parsing untuk memahami struktur kalimat:

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The cat sat on the mat.")

for token in doc:
    print(f"{token.text:<10} {token.dep_:<10} {token.head.text:<10}")
```

Output:

```
The        det        cat       
cat        nsubj      sat       
sat        ROOT       sat       
on         prep       sat       
the        det        mat       
mat        pobj       on        
.          punct      sat       
```

### Visualisasi Dependency Tree

```python
import spacy
from spacy import displacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The cat sat on the mat.")

# Render di Jupyter Notebook
displacy.render(doc, style="dep", jupyter=True)
```

## Menggabungkan Semua Komponen

```python
import spacy

def analyze_text(text, nlp):
    """Analisis lengkap teks dengan tokenisasi, POS, dan NER."""
    doc = nlp(text)
    
    print("=" * 60)
    print("TOKENISASI & POS TAGGING")
    print("=" * 60)
    print(f"{'Token':<15} {'Lemma':<15} {'POS':<8} {'Dep':<10}")
    print("-" * 60)
    
    for token in doc:
        print(f"{token.text:<15} {token.lemma_:<15} {token.pos_:<8} {token.dep_:<10}")
    
    print("\n" + "=" * 60)
    print("NAMED ENTITIES")
    print("=" * 60)
    
    if doc.ents:
        for ent in doc.ents:
            print(f"{ent.text:<25} {ent.label_:<10}")
    else:
        print("Tidak ada entitas terdeteksi.")
    
    print("\n" + "=" * 60)
    print("KALIMAT")
    print("=" * 60)
    
    for i, sent in enumerate(doc.sents):
        print(f"{i+1}. {sent.text}")

# Contoh penggunaan
nlp = spacy.load("en_core_web_sm")
text = "Barack Obama was born in Hawaii. He was the 44th President of the United States."
analyze_text(text, nlp)
```

## Ringkasan

| Komponen | Fungsi | Akses di spaCy |
|----------|--------|----------------|
| Tokenisasi | Memecah teks menjadi token | `for token in doc` |
| POS Tagging | Kategori gramatikal | `token.pos_`, `token.tag_` |
| NER | Identifikasi entitas | `doc.ents`, `ent.label_` |
| Dependency | Struktur kalimat | `token.dep_`, `token.head` |

## Langkah Selanjutnya

Anda telah mempelajari dasar-dasar NLP dengan spaCy. Untuk eksplorasi lebih lanjut:

- Pelajari [dokumentasi resmi spaCy](https://spacy.io/usage)
- Coba latih model kustom untuk domain spesifik
- Eksplorasi pustaka NLP lain seperti NLTK atau Hugging Face Transformers
