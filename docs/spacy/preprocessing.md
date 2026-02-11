# Preprocessing Teks

Preprocessing adalah langkah penting dalam NLP untuk membersihkan dan menyiapkan teks sebelum analisis lebih lanjut. Bagian ini membahas teknik-teknik preprocessing menggunakan spaCy.

## Tokenisasi

Tokenisasi adalah proses memecah teks menjadi unit-unit kecil (token):

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("I can't wait to see you! It's going to be great.")

for token in doc:
    print(token.text)
```

Output:

```
I
ca
n't
wait
to
see
you
!
It
's
going
to
be
great
.
```

:::{note}
spaCy secara otomatis memisahkan kontraksi seperti "can't" menjadi "ca" dan "n't", serta "It's" menjadi "It" dan "'s".
:::

## Lemmatisasi

Lemmatisasi mengubah kata ke bentuk dasarnya (lemma):

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The cats are running and jumping over the fences.")

for token in doc:
    if token.lemma_ != token.text:
        print(f"{token.text:12} → {token.lemma_}")
```

Output:

```
cats         → cat
are          → be
running      → run
jumping      → jump
fences       → fence
```

### Lemmatisasi Bahasa Indonesia

```python
import spacy

nlp = spacy.load("id_core_news_sm")
doc = nlp("Para mahasiswa sedang membaca buku-buku di perpustakaan.")

for token in doc:
    if token.lemma_ != token.text.lower():
        print(f"{token.text:15} → {token.lemma_}")
```

Output:

```
mahasiswa       → mahasiswa
membaca         → baca
buku-buku       → buku
perpustakaan    → pustaka
```

## Menghapus Stopwords

Stopwords adalah kata-kata umum yang sering tidak bermakna untuk analisis:

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("This is a sample sentence with some common words.")

# Filter stopwords
tokens_without_stopwords = [token for token in doc if not token.is_stop]

print("Dengan stopwords:")
print([token.text for token in doc])

print("\nTanpa stopwords:")
print([token.text for token in tokens_without_stopwords])
```

Output:

```
Dengan stopwords:
['This', 'is', 'a', 'sample', 'sentence', 'with', 'some', 'common', 'words', '.']

Tanpa stopwords:
['sample', 'sentence', 'common', 'words', '.']
```

### Melihat Daftar Stopwords

```python
import spacy

nlp = spacy.load("en_core_web_sm")

# Lihat beberapa stopwords
stopwords = list(nlp.Defaults.stop_words)[:20]
print(stopwords)
```

### Menambah/Menghapus Stopwords Kustom

```python
import spacy

nlp = spacy.load("en_core_web_sm")

# Menambah stopword kustom
nlp.Defaults.stop_words.add("sample")

# Menghapus stopword
nlp.Defaults.stop_words.discard("not")

# Atau update atribut is_stop pada vocab
nlp.vocab["sample"].is_stop = True
```

## Menghapus Tanda Baca

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Hello, world! How are you?")

# Filter tanda baca
tokens_without_punct = [token for token in doc if not token.is_punct]

print("Dengan tanda baca:")
print([token.text for token in doc])

print("\nTanpa tanda baca:")
print([token.text for token in tokens_without_punct])
```

Output:

```
Dengan tanda baca:
['Hello', ',', 'world', '!', 'How', 'are', 'you', '?']

Tanpa tanda baca:
['Hello', 'world', 'How', 'are', 'you']
```

## Normalisasi Teks

### Lowercase

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Python is AWESOME!")

# Menggunakan atribut lower_
for token in doc:
    print(f"{token.text:10} → {token.lower_}")
```

Output:

```
Python     → python
is         → is
AWESOME    → awesome
!          → !
```

### Menghapus Whitespace dan Karakter Khusus

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("This    has   extra   spaces   and\nnewlines!")

# Filter token yang bukan whitespace
tokens = [token.text for token in doc if not token.is_space]
print(tokens)
# ['This', 'has', 'extra', 'spaces', 'and', 'newlines', '!']
```

## Pipeline Preprocessing Lengkap

Berikut contoh fungsi preprocessing yang menggabungkan beberapa teknik:

```python
import spacy

def preprocess_text(text, nlp, remove_stopwords=True, remove_punct=True, lemmatize=True):
    """
    Preprocessing teks dengan spaCy.
    
    Args:
        text: Teks yang akan diproses
        nlp: Model spaCy yang sudah dimuat
        remove_stopwords: Hapus stopwords jika True
        remove_punct: Hapus tanda baca jika True
        lemmatize: Gunakan lemma jika True
    
    Returns:
        List of processed tokens
    """
    doc = nlp(text)
    
    tokens = []
    for token in doc:
        # Skip whitespace
        if token.is_space:
            continue
        
        # Skip stopwords jika diminta
        if remove_stopwords and token.is_stop:
            continue
        
        # Skip tanda baca jika diminta
        if remove_punct and token.is_punct:
            continue
        
        # Gunakan lemma atau text
        if lemmatize:
            tokens.append(token.lemma_.lower())
        else:
            tokens.append(token.lower_)
    
    return tokens

# Contoh penggunaan
nlp = spacy.load("en_core_web_sm")

text = "The quick brown foxes are jumping over the lazy dogs!"
result = preprocess_text(text, nlp)
print(result)
# ['quick', 'brown', 'fox', 'jump', 'lazy', 'dog']
```

## Contoh Preprocessing Bahasa Indonesia

```python
import spacy

def preprocess_indonesian(text, nlp):
    """Preprocessing untuk teks Bahasa Indonesia."""
    doc = nlp(text)
    
    tokens = []
    for token in doc:
        # Skip whitespace, stopwords, dan tanda baca
        if token.is_space or token.is_stop or token.is_punct:
            continue
        
        # Gunakan lemma
        tokens.append(token.lemma_.lower())
    
    return tokens

nlp = spacy.load("id_core_news_sm")

text = "Para mahasiswa sedang membaca buku-buku yang menarik di perpustakaan."
result = preprocess_indonesian(text, nlp)
print(result)
# ['mahasiswa', 'baca', 'buku', 'tarik', 'pustaka']
```

## Memproses Batch Dokumen

```python
import spacy

nlp = spacy.load("en_core_web_sm")

documents = [
    "The cats are sleeping on the couch.",
    "Dogs love to play in the park.",
    "Birds are flying in the blue sky."
]

# Preprocessing batch dengan nlp.pipe()
processed_docs = []
for doc in nlp.pipe(documents):
    tokens = [token.lemma_.lower() for token in doc 
              if not token.is_stop and not token.is_punct]
    processed_docs.append(tokens)

for i, tokens in enumerate(processed_docs):
    print(f"Doc {i+1}: {tokens}")
```

Output:

```
Doc 1: ['cat', 'sleep', 'couch']
Doc 2: ['dog', 'love', 'play', 'park']
Doc 3: ['bird', 'fly', 'blue', 'sky']
```

## Tips Preprocessing

:::{tip}
1. **Urutan preprocessing penting** - Biasanya tokenisasi → lowercasing → lemmatisasi → hapus stopwords
2. **Jangan selalu hapus stopwords** - Untuk beberapa tugas seperti sentiment analysis, stopwords bisa penting
3. **Pertimbangkan domain** - Preprocessing untuk teks medis berbeda dengan teks sosial media
4. **Simpan teks asli** - Selalu simpan teks asli sebelum preprocessing untuk referensi
:::

## Langkah Selanjutnya

Lanjutkan ke [Tokenisasi, POS & NER](tokenisasi_pos_ner.md) untuk mempelajari lebih dalam tentang analisis linguistik dengan spaCy.
