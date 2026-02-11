# Tensor

Tensor adalah struktur data fundamental di PyTorch, mirip dengan ndarray di NumPy tapi dengan kemampuan akselerasi GPU.

## Membuat Tensor

### Dari Data Langsung

```python
import torch

# Dari list
data = [[1, 2], [3, 4]]
tensor = torch.tensor(data)
print(tensor)
# tensor([[1, 2],
#         [3, 4]])

# Dari tuple
tensor = torch.tensor((1, 2, 3))
print(tensor)  # tensor([1, 2, 3])
```

### Dari NumPy

```python
import torch
import numpy as np

np_arr = np.array([1, 2, 3])

# Dari NumPy (shares memory)
tensor = torch.from_numpy(np_arr)
print(tensor)  # tensor([1, 2, 3])

# Ke NumPy
np_back = tensor.numpy()
print(np_back)  # [1 2 3]
```

### Dengan Fungsi Factory

```python
import torch

# Zeros
zeros = torch.zeros(3, 4)
print(zeros)

# Ones
ones = torch.ones(2, 3)
print(ones)

# Random
rand = torch.rand(2, 3)      # Uniform [0, 1)
randn = torch.randn(2, 3)    # Normal (0, 1)
randint = torch.randint(0, 10, (2, 3))  # Integer [0, 10)

# Seperti tensor lain
x = torch.ones(2, 3)
y = torch.zeros_like(x)
z = torch.rand_like(x)

# Range
arange = torch.arange(0, 10, 2)  # [0, 2, 4, 6, 8]
linspace = torch.linspace(0, 1, 5)  # [0, 0.25, 0.5, 0.75, 1]

# Identity
eye = torch.eye(3)
print(eye)
```

### Dengan requires_grad

```python
import torch

# Tensor dengan autograd enabled
autograd_tensor = torch.randn((2, 3, 4), requires_grad=True)
print(autograd_tensor.requires_grad)  # True
```

## Atribut Tensor

```python
import torch

tensor = torch.rand(3, 4)

print(f"Shape: {tensor.shape}")       # torch.Size([3, 4])
print(f"Size: {tensor.size()}")       # torch.Size([3, 4])
print(f"Dtype: {tensor.dtype}")       # torch.float32
print(f"Device: {tensor.device}")     # cpu
print(f"Requires grad: {tensor.requires_grad}")  # False
print(f"Ndim: {tensor.ndim}")         # 2
print(f"Numel: {tensor.numel()}")     # 12 (total elements)
```

## Tipe Data

```python
import torch

# Spesifikasi dtype saat membuat
tensor_int = torch.tensor([1, 2, 3], dtype=torch.int32)
tensor_float = torch.tensor([1, 2, 3], dtype=torch.float64)
tensor_bool = torch.tensor([True, False, True])

# Konversi tipe
x = torch.tensor([1.5, 2.5, 3.5])
x_int = x.int()       # atau x.to(torch.int32)
x_long = x.long()     # int64
x_float = x.float()   # float32
x_double = x.double() # float64

print(x_int)  # tensor([1, 2, 3], dtype=torch.int32)
```

## Indexing dan Slicing

```python
import torch

tensor = torch.arange(12).reshape(3, 4)
print(tensor)
# tensor([[ 0,  1,  2,  3],
#         [ 4,  5,  6,  7],
#         [ 8,  9, 10, 11]])

# Indexing
print(tensor[0, 0])      # tensor(0)
print(tensor[1, 2])      # tensor(6)
print(tensor[0, 0].item())  # 0 (Python scalar)

# Slicing
print(tensor[0, :])      # tensor([0, 1, 2, 3])
print(tensor[:, 0])      # tensor([0, 4, 8])
print(tensor[1:, 1:3])   # tensor([[5, 6], [9, 10]])

# Boolean indexing
mask = tensor > 5
print(tensor[mask])      # tensor([6, 7, 8, 9, 10, 11])
```

## Mengubah Bentuk

### reshape dan view

```python
import torch

x = torch.arange(12)
print(x.shape)  # torch.Size([12])

# Reshape
y = x.reshape(3, 4)
print(y.shape)  # torch.Size([3, 4])

# View (harus contiguous)
z = x.view(3, 4)
print(z.shape)  # torch.Size([3, 4])

# -1 untuk dimensi otomatis
a = x.reshape(-1, 3)
print(a.shape)  # torch.Size([4, 3])
```

### squeeze dan unsqueeze

```python
import torch

x = torch.zeros(2, 1, 3, 1, 4)
print(x.shape)  # torch.Size([2, 1, 3, 1, 4])

# Squeeze: hapus dimensi dengan ukuran 1
y = x.squeeze()
print(y.shape)  # torch.Size([2, 3, 4])

# Unsqueeze: tambah dimensi
z = torch.zeros(3, 4)
print(z.unsqueeze(0).shape)  # torch.Size([1, 3, 4])
print(z.unsqueeze(1).shape)  # torch.Size([3, 1, 4])
```

### permute dan transpose

```python
import torch

x = torch.zeros(2, 3, 4)

# Transpose (swap 2 dimensions)
y = x.transpose(0, 2)
print(y.shape)  # torch.Size([4, 3, 2])

# Permute (reorder all dimensions)
z = x.permute(2, 0, 1)
print(z.shape)  # torch.Size([4, 2, 3])
```

## Operasi Matematika

### Operasi Element-wise

```python
import torch

a = torch.tensor([1, 2, 3], dtype=torch.float32)
b = torch.tensor([4, 5, 6], dtype=torch.float32)

print(a + b)      # tensor([5., 7., 9.])
print(a - b)      # tensor([-3., -3., -3.])
print(a * b)      # tensor([4., 10., 18.])
print(a / b)      # tensor([0.25, 0.4, 0.5])
print(a ** 2)     # tensor([1., 4., 9.])
print(torch.sqrt(a))  # tensor([1., 1.4142, 1.7321])
```

### Operasi Matriks

```python
import torch

a = torch.randn(2, 3)
b = torch.randn(3, 4)

# Matrix multiplication
c = torch.matmul(a, b)  # atau a @ b
print(c.shape)  # torch.Size([2, 4])

# Dot product
x = torch.tensor([1, 2, 3], dtype=torch.float32)
y = torch.tensor([4, 5, 6], dtype=torch.float32)
print(torch.dot(x, y))  # tensor(32.)

# Element-wise multiplication
print(x * y)  # tensor([4., 10., 18.])
```

### Agregasi

```python
import torch

x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)

print(x.sum())        # tensor(21.)
print(x.mean())       # tensor(3.5)
print(x.max())        # tensor(6.)
print(x.min())        # tensor(1.)
print(x.std())        # tensor(1.8708)

# Per dimensi
print(x.sum(dim=0))   # tensor([5., 7., 9.])  - per kolom
print(x.sum(dim=1))   # tensor([6., 15.])     - per baris

# Dengan keepdim
print(x.sum(dim=1, keepdim=True))
# tensor([[ 6.],
#         [15.]])
```

## Concatenate dan Stack

```python
import torch

a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[5, 6], [7, 8]])

# Concatenate (along existing dimension)
cat0 = torch.cat([a, b], dim=0)
print(cat0.shape)  # torch.Size([4, 2])

cat1 = torch.cat([a, b], dim=1)
print(cat1.shape)  # torch.Size([2, 4])

# Stack (create new dimension)
stack = torch.stack([a, b], dim=0)
print(stack.shape)  # torch.Size([2, 2, 2])
```

## GPU Operations

```python
import torch

# Cek ketersediaan GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Buat tensor di GPU
if torch.cuda.is_available():
    x = torch.randn(3, 3, device="cuda")
    print(f"Tensor on: {x.device}")
    
    # Pindahkan ke GPU
    y = torch.randn(3, 3)
    y = y.to(device)
    print(f"Moved to: {y.device}")
    
    # Operasi di GPU
    z = x + y
    
    # Pindahkan kembali ke CPU
    z_cpu = z.cpu()
```

## In-place Operations

```python
import torch

x = torch.tensor([1, 2, 3], dtype=torch.float32)

# In-place operations (dengan underscore)
x.add_(1)      # x = x + 1
print(x)       # tensor([2., 3., 4.])

x.mul_(2)      # x = x * 2
print(x)       # tensor([4., 6., 8.])

x.zero_()      # x = 0
print(x)       # tensor([0., 0., 0.])
```

:::{warning}
In-place operations dapat menyebabkan masalah dengan autograd. Gunakan dengan hati-hati saat training neural networks.
:::

## Latihan

1. Buat tensor 3D random dan lakukan operasi reshape
2. Konversi array NumPy ke tensor PyTorch dan sebaliknya
3. Lakukan matrix multiplication pada GPU (jika tersedia)
4. Implementasikan normalization menggunakan operasi tensor
