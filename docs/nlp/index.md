# Natural Language Processing

Natural Language Processing (NLP) adalah cabang kecerdasan buatan yang berfokus pada interaksi antara komputer dan bahasa manusia. Dengan NLP, komputer dapat memahami, menganalisis, dan menghasilkan teks dalam bahasa alami.

## Apa itu NLP?

NLP menggabungkan linguistik komputasional dengan machine learning dan deep learning untuk memproses bahasa manusia. Beberapa tugas umum dalam NLP meliputi:

- **Tokenisasi** - Memecah teks menjadi unit-unit kecil (token)
- **Part-of-Speech Tagging** - Menandai kategori gramatikal setiap kata
- **Named Entity Recognition** - Mengidentifikasi entitas seperti nama orang, lokasi, organisasi
- **Sentiment Analysis** - Menentukan sentimen (positif/negatif) dari teks
- **Text Classification** - Mengklasifikasikan teks ke dalam kategori
- **Machine Translation** - Menerjemahkan teks antar bahasa
- **Question Answering** - Menjawab pertanyaan berdasarkan konteks
- **Text Summarization** - Meringkas teks panjang

## Pustaka NLP

Dalam materi ini, kita akan mempelajari dua pustaka NLP populer:

### spaCy

Pustaka NLP yang cepat dan efisien untuk penggunaan produksi. Cocok untuk:
- Pemrosesan teks real-time
- Pipeline NLP lengkap (tokenisasi, POS, NER, parsing)
- Aplikasi yang membutuhkan kecepatan tinggi

### Hugging Face Transformers

Pustaka yang menyediakan akses ke model-model state-of-the-art. Cocok untuk:
- Tugas yang membutuhkan akurasi maksimal
- Menggunakan model pre-trained canggih (BERT, GPT, dll)
- Eksperimen dan penelitian

## Daftar Materi

```{toctree}
:maxdepth: 2

spacy/index
transformers/index
```

## Alur Pembelajaran

```
┌─────────────────────────────────────────────────────┐
│                   NLP Dasar                         │
│         (Konsep: tokenisasi, POS, NER)              │
└───────────────────────┬─────────────────────────────┘
                        │
          ┌─────────────┴─────────────┐
          ▼                           ▼
┌─────────────────┐         ┌─────────────────┐
│     spaCy       │         │  Transformers   │
│  (Cepat, Prod)  │         │ (State-of-art)  │
└─────────────────┘         └─────────────────┘
```

## Langkah Selanjutnya

Mulai dengan [spaCy](spacy/index.md) untuk mempelajari dasar-dasar NLP dengan pustaka yang cepat dan efisien, atau langsung ke [Transformers](transformers/index.md) jika Anda ingin menggunakan model-model canggih.
