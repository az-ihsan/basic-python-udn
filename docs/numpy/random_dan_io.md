# Random dan I/O

NumPy menyediakan generator bilangan random yang powerful dan fungsi untuk membaca/menulis array ke file.

## Bilangan Random

### Random Dasar

```python
import numpy as np

# Set seed untuk reprodusibilitas
np.random.seed(42)

# Random float antara 0 dan 1
print(np.random.random())       # Single value
print(np.random.random(5))      # Array of 5
print(np.random.random((2, 3))) # 2x3 array

# Random integer
print(np.random.randint(1, 10))         # 1 sampai 9
print(np.random.randint(1, 10, size=5)) # Array of 5
print(np.random.randint(1, 10, size=(2, 3))) # 2x3 array
```

### Distribusi Normal

```python
import numpy as np

np.random.seed(42)

# Standard normal (mean=0, std=1)
print(np.random.randn(5))

# Normal dengan mean dan std tertentu
mean = 100
std = 15
print(np.random.normal(mean, std, size=5))

# Contoh: simulasi skor IQ
skor_iq = np.random.normal(100, 15, size=1000)
print(f"Mean: {skor_iq.mean():.2f}")
print(f"Std: {skor_iq.std():.2f}")
```

### Distribusi Lainnya

```python
import numpy as np

np.random.seed(42)

# Uniform distribution
uniform = np.random.uniform(low=0, high=10, size=5)
print(uniform)

# Binomial distribution
binomial = np.random.binomial(n=10, p=0.5, size=5)
print(binomial)

# Poisson distribution
poisson = np.random.poisson(lam=5, size=5)
print(poisson)

# Exponential distribution
exponential = np.random.exponential(scale=2, size=5)
print(exponential)
```

### Sampling

```python
import numpy as np

np.random.seed(42)

data = np.array([10, 20, 30, 40, 50])

# Pilih elemen random
print(np.random.choice(data))           # Satu elemen
print(np.random.choice(data, size=3))   # Dengan replacement
print(np.random.choice(data, size=3, replace=False))  # Tanpa replacement

# Dengan probabilitas
probs = [0.1, 0.1, 0.1, 0.3, 0.4]
print(np.random.choice(data, size=10, p=probs))

# Shuffle array (in-place)
arr = np.array([1, 2, 3, 4, 5])
np.random.shuffle(arr)
print(arr)  # Urutan teracak

# Permutasi (return copy)
arr = np.array([1, 2, 3, 4, 5])
perm = np.random.permutation(arr)
print(perm)  # Array teracak
print(arr)   # Original tidak berubah
```

### Generator Baru (NumPy 1.17+)

```python
import numpy as np

# Generator lebih modern dan fleksibel
rng = np.random.default_rng(seed=42)

print(rng.random(5))
print(rng.integers(1, 10, size=5))
print(rng.normal(0, 1, size=5))
print(rng.choice([1, 2, 3, 4, 5], size=3))
```

## File I/O

### Menyimpan dan Membaca Array (.npy)

```python
import numpy as np

arr = np.array([[1, 2, 3], [4, 5, 6]])

# Simpan ke file .npy
np.save('array.npy', arr)

# Baca dari file
loaded = np.load('array.npy')
print(loaded)
# [[1 2 3]
#  [4 5 6]]
```

### Multiple Arrays (.npz)

```python
import numpy as np

arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])

# Simpan beberapa array
np.savez('arrays.npz', x=arr1, y=arr2)

# Baca
data = np.load('arrays.npz')
print(data['x'])  # [1 2 3]
print(data['y'])  # [4 5 6]

# Simpan dengan kompresi
np.savez_compressed('arrays_compressed.npz', x=arr1, y=arr2)
```

### Text Files

```python
import numpy as np

arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Simpan ke text file
np.savetxt('array.txt', arr, delimiter=',', fmt='%d')
# Isi file:
# 1,2,3
# 4,5,6
# 7,8,9

# Baca dari text file
loaded = np.loadtxt('array.txt', delimiter=',')
print(loaded)

# Dengan header
np.savetxt('array_header.csv', arr, 
           delimiter=',', 
           fmt='%d',
           header='A,B,C',
           comments='')

# Baca dengan skip header
loaded = np.loadtxt('array_header.csv', 
                     delimiter=',', 
                     skiprows=1)
```

### Membaca CSV dengan genfromtxt

```python
import numpy as np

# genfromtxt lebih fleksibel untuk data dengan missing values
# Contoh file data.csv:
# nama,nilai1,nilai2
# Ahmad,85,90
# Budi,,78
# Citra,92,88

data = np.genfromtxt('data.csv', 
                      delimiter=',',
                      skip_header=1,
                      usecols=(1, 2),
                      filling_values=0)
print(data)
# [[85. 90.]
#  [ 0. 78.]
#  [92. 88.]]
```

## Contoh Praktis

### Simulasi Monte Carlo

```python
import numpy as np

def estimate_pi(n_samples):
    """Estimasi nilai pi dengan Monte Carlo"""
    rng = np.random.default_rng(42)
    
    # Generate titik random dalam kotak [-1, 1] x [-1, 1]
    x = rng.uniform(-1, 1, n_samples)
    y = rng.uniform(-1, 1, n_samples)
    
    # Hitung titik di dalam lingkaran
    inside = np.sum(x**2 + y**2 <= 1)
    
    # pi/4 = rasio area lingkaran/kotak
    return 4 * inside / n_samples

for n in [100, 1000, 10000, 100000]:
    print(f"n={n:6d}: pi ≈ {estimate_pi(n):.6f}")
```

### Bootstrap Sampling

```python
import numpy as np

def bootstrap_mean(data, n_bootstrap=1000, confidence=0.95):
    """Hitung confidence interval untuk mean"""
    rng = np.random.default_rng(42)
    n = len(data)
    
    # Generate bootstrap samples
    bootstrap_means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(data, size=n, replace=True)
        bootstrap_means[i] = np.mean(sample)
    
    # Hitung confidence interval
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
    
    return np.mean(bootstrap_means), lower, upper

# Contoh penggunaan
data = np.random.normal(100, 15, 50)
mean, lower, upper = bootstrap_mean(data)
print(f"Mean: {mean:.2f} ({lower:.2f} - {upper:.2f})")
```

### Simpan Model Sederhana

```python
import numpy as np

# Simpan parameter model
weights = np.random.randn(100, 50)
biases = np.random.randn(50)

np.savez('model.npz', 
         weights=weights, 
         biases=biases,
         metadata={'version': '1.0', 'date': '2026-02-11'})

# Load model
model = np.load('model.npz', allow_pickle=True)
loaded_weights = model['weights']
loaded_biases = model['biases']
loaded_metadata = model['metadata'].item()

print(f"Weights shape: {loaded_weights.shape}")
print(f"Model version: {loaded_metadata['version']}")
```

### Membaca Data Besar dengan memmap

```python
import numpy as np

# Untuk file yang sangat besar, gunakan memory mapping
# Ini tidak memuat seluruh file ke memori

# Buat file besar
big_arr = np.random.random((10000, 1000))
np.save('big_array.npy', big_arr)

# Baca dengan memmap (hanya load bagian yang diakses)
mmap = np.load('big_array.npy', mmap_mode='r')
print(mmap.shape)  # (10000, 1000)
print(mmap[0:10, 0:5])  # Hanya load bagian ini ke memori
```

## Latihan

1. Generate 1000 sampel dari distribusi normal dan plot histogramnya
2. Implementasikan simulasi random walk 2D
3. Simpan dan load dataset dengan metadata menggunakan npz
4. Implementasikan cross-validation split menggunakan random sampling
