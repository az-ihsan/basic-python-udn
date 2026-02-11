# spaCy Dasar

Bagian ini membahas konsep dasar spaCy: cara memuat model, objek-objek utama seperti `Doc`, `Token`, dan `Span`, serta atribut-atribut penting yang dapat diakses.

## Memuat Model

```python
import spacy

# Load model bahasa Inggris
nlp_en = spacy.load("en_core_web_sm")

# Load model bahasa Indonesia
nlp_id = spacy.load("id_core_news_sm")
```

:::{tip}
Variabel `nlp` adalah konvensi penamaan yang umum digunakan untuk objek Language di spaCy.
:::

## Objek Doc

Ketika Anda memproses teks, spaCy mengembalikan objek `Doc` yang berisi semua informasi hasil analisis:

```python
import spacy

nlp = spacy.load("en_core_web_sm")

# Memproses teks menghasilkan objek Doc
doc = nlp("This is a sentence.")

# Doc adalah sequence dari Token
print(type(doc))  # <class 'spacy.tokens.doc.Doc'>
print(len(doc))   # 5 (jumlah token)
```

### Mengakses Token

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Hello world!")

# Iterasi token
for token in doc:
    print(token.text)

# Akses token dengan indeks
first_token = doc[0]
print(first_token.text)  # Hello

# Slicing
tokens = doc[0:2]
print(tokens.text)  # Hello world
```

## Objek Token

Setiap token memiliki banyak atribut yang berguna:

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple is looking at buying a U.K. startup.")

for token in doc:
    print(f"Text: {token.text:10} | "
          f"Lemma: {token.lemma_:10} | "
          f"POS: {token.pos_:6} | "
          f"Dep: {token.dep_:10} | "
          f"Shape: {token.shape_}")
```

Output:

```
Text: Apple      | Lemma: Apple      | POS: PROPN  | Dep: nsubj      | Shape: Xxxxx
Text: is         | Lemma: be         | POS: AUX    | Dep: aux        | Shape: xx
Text: looking    | Lemma: look       | POS: VERB   | Dep: ROOT       | Shape: xxxx
Text: at         | Lemma: at         | POS: ADP    | Dep: prep       | Shape: xx
Text: buying     | Lemma: buy        | POS: VERB   | Dep: pcomp      | Shape: xxxx
Text: a          | Lemma: a          | POS: DET    | Dep: det        | Shape: x
Text: U.K.       | Lemma: U.K.       | POS: PROPN  | Dep: nmod       | Shape: X.X.
Text: startup    | Lemma: startup    | POS: NOUN   | Dep: dobj       | Shape: xxxx
Text: .          | Lemma: .          | POS: PUNCT  | Dep: punct      | Shape: .
```

### Atribut Token Penting

| Atribut | Deskripsi |
|---------|-----------|
| `text` | Teks asli token |
| `lemma_` | Bentuk dasar kata |
| `pos_` | Part-of-speech tag (coarse) |
| `tag_` | Part-of-speech tag (fine-grained) |
| `dep_` | Dependency relation |
| `shape_` | Pola bentuk kata (Xxxxx, xxxx, dll) |
| `is_alpha` | Apakah token adalah huruf? |
| `is_stop` | Apakah token adalah stopword? |
| `is_punct` | Apakah token adalah tanda baca? |
| `is_digit` | Apakah token adalah angka? |

### Contoh Penggunaan Atribut Boolean

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("I have 3 cats, and they're cute!")

for token in doc:
    print(f"{token.text:10} | "
          f"Alpha: {token.is_alpha} | "
          f"Stop: {token.is_stop} | "
          f"Punct: {token.is_punct}")
```

Output:

```
I          | Alpha: True  | Stop: True  | Punct: False
have       | Alpha: True  | Stop: True  | Punct: False
3          | Alpha: False | Stop: False | Punct: False
cats       | Alpha: True  | Stop: False | Punct: False
,          | Alpha: False | Stop: False | Punct: True
and        | Alpha: True  | Stop: True  | Punct: False
they       | Alpha: True  | Stop: True  | Punct: False
're        | Alpha: True  | Stop: True  | Punct: False
cute       | Alpha: True  | Stop: False | Punct: False
!          | Alpha: False | Stop: False | Punct: True
```

## Objek Span

`Span` adalah potongan dari `Doc`, seperti frasa atau kalimat:

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Hello world! How are you?")

# Membuat span dengan slicing
span = doc[0:2]  # "Hello world"
print(span.text)

# Mengakses kalimat
for sent in doc.sents:
    print(f"Sentence: {sent.text}")
```

Output:

```
Hello world
Sentence: Hello world!
Sentence: How are you?
```

## Contoh dengan Bahasa Indonesia

```python
import spacy

nlp = spacy.load("id_core_news_sm")
doc = nlp("Saya sedang belajar pemrograman Python di universitas.")

# Lihat atribut token
for token in doc:
    print(f"{token.text:15} | "
          f"Lemma: {token.lemma_:15} | "
          f"POS: {token.pos_:6}")
```

Output:

```
Saya            | Lemma: saya            | POS: PRON  
sedang          | Lemma: sedang          | POS: ADV   
belajar         | Lemma: ajar            | POS: VERB  
pemrograman     | Lemma: program         | POS: NOUN  
Python          | Lemma: Python          | POS: PROPN 
di              | Lemma: di              | POS: ADP   
universitas     | Lemma: universitas     | POS: NOUN  
.               | Lemma: .               | POS: PUNCT 
```

## Melihat Komponen Pipeline

Anda dapat melihat komponen apa saja yang ada di pipeline model:

```python
import spacy

nlp = spacy.load("en_core_web_sm")

# Lihat nama komponen pipeline
print(nlp.pipe_names)
# ['tok2vec', 'tagger', 'parser', 'attribute_ruler', 'lemmatizer', 'ner']

# Lihat detail pipeline
print(nlp.pipeline)
```

## Memproses Banyak Teks

Untuk efisiensi, gunakan `nlp.pipe()` saat memproses banyak teks:

```python
import spacy

nlp = spacy.load("en_core_web_sm")

texts = [
    "This is the first document.",
    "This is the second document.",
    "And this is the third one."
]

# Lebih efisien daripada loop biasa
for doc in nlp.pipe(texts):
    print(f"Tokens: {len(doc)} | First token: {doc[0].text}")
```

Output:

```
Tokens: 6 | First token: This
Tokens: 6 | First token: This
Tokens: 7 | First token: And
```

## Langkah Selanjutnya

Lanjutkan ke [Preprocessing](preprocessing.md) untuk mempelajari teknik pembersihan dan normalisasi teks.
