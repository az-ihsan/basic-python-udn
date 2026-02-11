# Machine Learning & Deep Learning

Machine Learning (ML) adalah cabang kecerdasan buatan yang membuat komputer belajar pola dari data. Deep Learning (DL) adalah subset dari ML yang memakai jaringan saraf berlapis untuk memodelkan pola yang kompleks.

## Apa itu ML & DL?

ML dan DL digunakan untuk memecahkan berbagai masalah berbasis data, seperti:

- **Klasifikasi** - Memprediksi label kategori (misal: spam vs bukan spam)
- **Regresi** - Memprediksi nilai kontinu (misal: harga rumah)
- **Clustering** - Mengelompokkan data tanpa label
- **Reduksi Dimensi** - Meringkas fitur agar lebih ringkas
- **Deteksi Anomali** - Menemukan pola yang menyimpang

## Konsep Kunci

- **Data dan fitur** - Representasi numerik dari informasi
- **Model dan parameter** - Fungsi yang dipelajari dari data
- **Loss function** - Mengukur kesalahan prediksi
- **Optimisasi** - Meminimalkan loss (misal: gradient descent)
- **Evaluasi** - Mengukur performa dan generalisasi

## Matematika Machine Learning

Pada machine learning klasik, kita memiliki data pelatihan berupa pasangan input-output:

$$
\mathcal{D} = \{(x_1, y_1), (x_2, y_2), \ldots, (x_n, y_n)\}
$$

di mana $x_i \in \mathbb{R}^d$ adalah vektor fitur dan $y_i$ adalah label atau nilai target.

### Model dan Prediksi

Model memetakan input ke prediksi melalui parameter $w$ (weights) dan $b$ (bias):

$$
\hat{y} = f(x; w, b)
$$

Untuk **regresi linear**, modelnya adalah:

$$
\hat{y} = w^\top x + b = \sum_{j=1}^{d} w_j x_j + b
$$

Untuk **regresi logistik** (klasifikasi), kita menambahkan fungsi sigmoid:

$$
\hat{y} = \sigma(w^\top x + b) = \frac{1}{1 + e^{-(w^\top x + b)}}
$$

### Loss Function

Loss function mengukur seberapa jauh prediksi dari nilai sebenarnya. Tujuan training adalah meminimalkan total loss pada seluruh data.

**Mean Squared Error (MSE)** untuk regresi:

$$
L_{\text{MSE}} = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)^2
$$

**Cross-Entropy Loss** untuk klasifikasi biner:

$$
L_{\text{CE}} = -\frac{1}{n} \sum_{i=1}^{n} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]
$$

**Hinge Loss** untuk Support Vector Machine (SVM):

$$
L_{\text{hinge}} = \frac{1}{n} \sum_{i=1}^{n} \max(0, 1 - y_i \cdot (w^\top x_i))
$$

di mana $y_i \in \{-1, +1\}$.

### Regularisasi

Untuk mencegah overfitting, kita menambahkan penalti pada parameter. **Ridge regression** (L2 regularization) meminimalkan:

$$
\min_{w} \lVert Xw - y \rVert_2^2 + \alpha \lVert w \rVert_2^2
$$

di mana $\alpha \geq 0$ mengontrol kekuatan regularisasi. Semakin besar $\alpha$, parameter cenderung lebih kecil.

### Gradient Descent

Untuk meminimalkan loss, kita update parameter secara iteratif menggunakan gradient descent:

$$
w \leftarrow w - \eta \frac{\partial L}{\partial w}
$$

di mana $\eta$ adalah **learning rate** yang mengontrol seberapa besar langkah update. Proses ini diulang hingga loss konvergen.

---

## Matematika Deep Learning

Deep learning menggunakan **neural network** yang terdiri dari banyak layer. Setiap layer melakukan transformasi linear diikuti fungsi aktivasi non-linear.

### Forward Propagation

Misalkan kita punya network dengan 2 hidden layer:

**Layer 1:**
$$
z^{(1)} = W^{(1)} x + b^{(1)}, \quad a^{(1)} = \sigma(z^{(1)})
$$

**Layer 2:**
$$
z^{(2)} = W^{(2)} a^{(1)} + b^{(2)}, \quad a^{(2)} = \sigma(z^{(2)})
$$

**Output:**
$$
\hat{y} = W^{(3)} a^{(2)} + b^{(3)}
$$

di mana $W^{(l)}$ adalah matriks bobot layer ke-$l$, $b^{(l)}$ adalah bias, dan $\sigma$ adalah fungsi aktivasi (misal ReLU, sigmoid, atau tanh).

### Fungsi Aktivasi

Beberapa fungsi aktivasi yang umum digunakan:

**ReLU (Rectified Linear Unit):**
$$
\text{ReLU}(z) = \max(0, z)
$$

**Sigmoid:**
$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

**Tanh:**
$$
\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}
$$

### Backward Propagation (Autograd)

Untuk menghitung gradien loss terhadap setiap parameter, kita menggunakan **chain rule**. PyTorch melakukan ini secara otomatis melalui sistem **autograd**.

Contoh sederhana: misalkan kita punya fungsi

$$
Q = 3a^3 - b^2
$$

Gradien terhadap parameter:

$$
\frac{\partial Q}{\partial a} = 9a^2
$$

$$
\frac{\partial Q}{\partial b} = -2b
$$

Dalam PyTorch, kita cukup memanggil `Q.backward()` dan gradien akan dihitung otomatis.

### Training Loop

Proses training neural network:

1. **Forward pass**: hitung prediksi $\hat{y} = f(x; \theta)$
2. **Compute loss**: hitung $L = \ell(\hat{y}, y)$
3. **Backward pass**: hitung gradien $\nabla_\theta L$ menggunakan backpropagation
4. **Update parameters**: $\theta \leftarrow \theta - \eta \nabla_\theta L$
5. **Ulangi** untuk semua batch dan epoch

### Optimizer

Selain gradient descent sederhana, ada beberapa optimizer yang lebih canggih:

**SGD dengan Momentum:**
$$
v_t = \gamma v_{t-1} + \eta \nabla_\theta L
$$
$$
\theta \leftarrow \theta - v_t
$$

**Adam** menggabungkan momentum dengan adaptive learning rate untuk konvergensi yang lebih cepat dan stabil.

## Pustaka Utama

Dalam materi ini, kita akan memakai dua pustaka utama:

### Scikit-learn

Pustaka ML klasik dengan API yang konsisten dan mudah dipakai. Cocok untuk:

- Model klasik (regresi, klasifikasi, clustering)
- Pipeline preprocessing dan evaluasi
- Baseline cepat untuk eksperimen

### PyTorch

Framework DL yang fleksibel dan populer untuk riset serta produksi. Cocok untuk:

- Neural network dan deep learning
- Eksperimen cepat dengan graph dinamis
- Training loop kustom

## Alur Pembelajaran

```
┌─────────────────────────────────────────────────────┐
│                ML & DL Dasar                        │
│   (fitur, model, loss, optimisasi, evaluasi)         │
└───────────────────────┬─────────────────────────────┘
                        │
          ┌─────────────┴─────────────┐
          ▼                           ▼
┌─────────────────┐         ┌─────────────────┐
│  Scikit-learn   │         │     PyTorch     │
│   (ML klasik)   │         │ (Deep Learning) │
└─────────────────┘         └─────────────────┘
```

## Langkah Selanjutnya

Mulai dengan [Scikit-learn](../scikit_learn/index.md) untuk memahami ML klasik, atau lanjutkan ke [PyTorch](../pytorch/index.md) untuk deep learning.

## Referensi

- Ridge regression: https://scikit-learn.org/stable/modules/linear_model.html#ridge-regression
- Hinge loss: https://scikit-learn.org/stable/modules/model_evaluation.html#hinge-loss
- PyTorch autograd tutorial: https://raw.githubusercontent.com/pytorch/tutorials/refs/heads/main/beginner_source/blitz/autograd_tutorial.py
