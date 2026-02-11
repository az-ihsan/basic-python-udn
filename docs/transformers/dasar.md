# Transformers Dasar

Hugging Face Transformers adalah pustaka yang menyediakan akses mudah ke ribuan model pre-trained untuk Natural Language Processing. Dengan Transformers, Anda dapat melakukan berbagai tugas NLP hanya dengan beberapa baris kode.

## Apa itu Transformers?

Transformers adalah arsitektur neural network yang menjadi dasar model-model modern seperti BERT, GPT, dan T5. Pustaka Hugging Face Transformers memudahkan penggunaan model-model ini tanpa perlu melatih dari awal.

**Keunggulan Transformers:**

- **Pre-trained models** - Model sudah dilatih pada data besar, tinggal pakai
- **State-of-the-art** - Performa terbaik untuk banyak tugas NLP
- **Mudah digunakan** - API `pipeline` yang sangat sederhana
- **Banyak pilihan** - Ribuan model untuk berbagai bahasa dan tugas
- **Komunitas besar** - Hugging Face Hub dengan model dari seluruh dunia

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

## Konsep Utama

### 1. Pipeline

`pipeline` adalah cara termudah untuk menggunakan model. Anda cukup tentukan tugas yang ingin dilakukan:

```python
from transformers import pipeline

# Buat pipeline untuk analisis sentimen
classifier = pipeline("sentiment-analysis")

# Gunakan pipeline
result = classifier("I love learning about AI!")
print(result)
# [{'label': 'POSITIVE', 'score': 0.9998}]
```

### 2. Tokenizer

Tokenizer mengubah teks menjadi format yang dipahami model:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Tokenisasi teks
tokens = tokenizer("Hello, how are you?")
print(tokens)
# {'input_ids': [101, 7592, 1010, 2129, 2024, 2017, 1029, 102], 
#  'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1]}

# Decode kembali ke teks
text = tokenizer.decode(tokens["input_ids"])
print(text)
# [CLS] hello, how are you? [SEP]
```

### 3. Model

Model melakukan prediksi berdasarkan input yang sudah di-tokenisasi:

```python
from transformers import AutoModelForSequenceClassification
import torch

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english"
)

# Input sudah di-tokenisasi
inputs = tokenizer("I love this!", return_tensors="pt")

# Prediksi
with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    print(predictions)
```

## Pipeline untuk Berbagai Tugas

### Analisis Sentimen

```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")

texts = [
    "This movie is amazing!",
    "I really hate waiting in long lines.",
    "The weather is okay today."
]

results = classifier(texts)
for text, result in zip(texts, results):
    print(f"{text}")
    print(f"  → {result['label']} ({result['score']:.2%})\n")
```

Output:

```
This movie is amazing!
  → POSITIVE (99.98%)

I really hate waiting in long lines.
  → NEGATIVE (99.97%)

The weather is okay today.
  → POSITIVE (99.55%)
```

### Named Entity Recognition (NER)

```python
from transformers import pipeline

ner = pipeline("ner", aggregation_strategy="simple")

text = "Apple Inc. was founded by Steve Jobs in California."
entities = ner(text)

for entity in entities:
    print(f"{entity['word']:20} → {entity['entity_group']:10} ({entity['score']:.2%})")
```

Output:

```
Apple Inc.           → ORG        (99.89%)
Steve Jobs           → PER        (99.93%)
California           → LOC        (99.96%)
```

### Text Generation

```python
from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")

prompt = "Artificial intelligence is"
result = generator(prompt, max_length=50, num_return_sequences=1)

print(result[0]["generated_text"])
```

### Question Answering

```python
from transformers import pipeline

qa = pipeline("question-answering")

context = """
Python is a programming language created by Guido van Rossum. 
It was first released in 1991 and is known for its simple syntax.
"""

question = "Who created Python?"
result = qa(question=question, context=context)

print(f"Answer: {result['answer']}")
print(f"Score: {result['score']:.2%}")
```

Output:

```
Answer: Guido van Rossum
Score: 98.45%
```

### Text Summarization

```python
from transformers import pipeline

summarizer = pipeline("summarization")

article = """
Machine learning is a subset of artificial intelligence that enables 
systems to learn and improve from experience without being explicitly 
programmed. It focuses on developing algorithms that can access data 
and use it to learn for themselves. The process begins with observations 
or data, such as examples, direct experience, or instruction, in order 
to look for patterns in data and make better decisions in the future.
"""

summary = summarizer(article, max_length=50, min_length=20)
print(summary[0]["summary_text"])
```

### Translation

```python
from transformers import pipeline

translator = pipeline("translation_en_to_de")

text = "Hello, how are you today?"
result = translator(text)

print(f"English: {text}")
print(f"German: {result[0]['translation_text']}")
```

## Menggunakan Model Multibahasa

Untuk bahasa Indonesia, Anda dapat menggunakan model multibahasa:

### Sentiment Analysis Bahasa Indonesia

```python
from transformers import pipeline

# Menggunakan model multibahasa
classifier = pipeline(
    "sentiment-analysis",
    model="nlptown/bert-base-multilingual-uncased-sentiment"
)

texts_id = [
    "Film ini sangat bagus dan menghibur!",
    "Pelayanannya buruk sekali.",
    "Makanannya biasa saja."
]

for text in texts_id:
    result = classifier(text)[0]
    print(f"{text}")
    print(f"  → {result['label']} ({result['score']:.2%})\n")
```

### NER Bahasa Indonesia

```python
from transformers import pipeline

# Model NER multibahasa
ner = pipeline(
    "ner",
    model="Davlan/xlm-roberta-base-ner-hrl",
    aggregation_strategy="simple"
)

text = "Presiden Joko Widodo mengunjungi Jakarta pada hari Senin."
entities = ner(text)

for entity in entities:
    print(f"{entity['word']:20} → {entity['entity_group']}")
```

Output:

```
Joko Widodo          → PER
Jakarta              → LOC
```

## Tips untuk Pemula

:::{tip}
1. **Mulai dengan `pipeline`** - Cara termudah untuk mencoba berbagai tugas NLP
2. **Pilih model yang tepat** - Kunjungi [Hugging Face Hub](https://huggingface.co/models) untuk mencari model
3. **Perhatikan ukuran model** - Model besar lebih akurat tapi membutuhkan lebih banyak memori
4. **Gunakan GPU jika ada** - Transformers otomatis menggunakan GPU jika tersedia
5. **Cache model** - Model diunduh sekali dan disimpan di cache untuk penggunaan berikutnya
:::

## Daftar Pipeline yang Tersedia

| Pipeline | Tugas | Contoh Penggunaan |
|----------|-------|-------------------|
| `sentiment-analysis` | Analisis sentimen | Positif/Negatif |
| `ner` | Named Entity Recognition | Deteksi nama, lokasi |
| `question-answering` | Menjawab pertanyaan | QA dari konteks |
| `summarization` | Meringkas teks | Ringkasan artikel |
| `translation_xx_to_yy` | Terjemahan | EN ke DE, FR, dll |
| `text-generation` | Generasi teks | Melanjutkan kalimat |
| `fill-mask` | Isi kata kosong | Prediksi kata |
| `zero-shot-classification` | Klasifikasi tanpa training | Kategori custom |

## Contoh: Zero-Shot Classification

Klasifikasi teks tanpa perlu melatih model untuk kategori tertentu:

```python
from transformers import pipeline

classifier = pipeline("zero-shot-classification")

text = "The new iPhone features an amazing camera and long battery life."

# Tentukan kategori yang diinginkan
candidate_labels = ["technology", "sports", "politics", "entertainment"]

result = classifier(text, candidate_labels)

for label, score in zip(result["labels"], result["scores"]):
    print(f"{label:15} → {score:.2%}")
```

Output:

```
technology      → 97.23%
entertainment   → 1.45%
politics        → 0.72%
sports          → 0.60%
```

## Perbandingan spaCy vs Transformers

| Aspek | spaCy | Transformers |
|-------|-------|--------------|
| Kecepatan | Sangat cepat | Lebih lambat |
| Akurasi | Baik | State-of-the-art |
| Ukuran model | Kecil (MB) | Besar (100MB-GB) |
| GPU | Opsional | Disarankan |
| Kasus penggunaan | Produksi, real-time | Akurasi maksimal |
| Pembelajaran | Mudah dipelajari | Butuh pemahaman lebih |

:::{note}
spaCy dan Transformers bisa digunakan bersama! spaCy mendukung integrasi dengan model Transformers melalui `spacy-transformers`.
:::

## Langkah Selanjutnya

- Eksplorasi [Hugging Face Hub](https://huggingface.co/models) untuk menemukan model yang sesuai
- Pelajari [dokumentasi resmi](https://huggingface.co/docs/transformers) untuk fitur lanjutan
- Coba fine-tuning model untuk tugas spesifik Anda
