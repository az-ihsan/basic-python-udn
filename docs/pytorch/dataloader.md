# DataLoader

DataLoader adalah utility di PyTorch untuk memuat data dalam batch, melakukan shuffling, dan parallel loading.

## Dataset Class

### Custom Dataset

```python
import torch
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Contoh penggunaan
import numpy as np

X = np.random.randn(100, 10)
y = np.random.randint(0, 2, 100)

dataset = CustomDataset(X, y)
print(f"Dataset size: {len(dataset)}")
print(f"Sample: {dataset[0]}")
```

### TensorDataset

```python
import torch
from torch.utils.data import TensorDataset, DataLoader

X = torch.randn(100, 10)
y = torch.randint(0, 2, (100,))

dataset = TensorDataset(X, y)
print(f"Dataset size: {len(dataset)}")
```

## DataLoader

### Basic Usage

```python
from torch.utils.data import DataLoader

dataloader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True
)

# Iterasi
for batch_X, batch_y in dataloader:
    print(f"Batch X shape: {batch_X.shape}")
    print(f"Batch y shape: {batch_y.shape}")
    break
```

### Parameter Penting

```python
from torch.utils.data import DataLoader

dataloader = DataLoader(
    dataset,
    batch_size=32,         # Ukuran batch
    shuffle=True,          # Acak urutan
    num_workers=4,         # Parallel loading
    drop_last=True,        # Buang batch terakhir jika tidak penuh
    pin_memory=True        # Untuk GPU
)
```

## Split Train/Val/Test

```python
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

# Buat dataset
X = torch.randn(1000, 10)
y = torch.randint(0, 3, (1000,))
dataset = TensorDataset(X, y)

# Split
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    dataset, [train_size, val_size, test_size]
)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"Train: {len(train_dataset)}")
print(f"Val: {len(val_dataset)}")
print(f"Test: {len(test_dataset)}")
```

## Custom Collate Function

```python
import torch
from torch.utils.data import DataLoader

def custom_collate(batch):
    """Custom function untuk menggabungkan samples menjadi batch."""
    X = torch.stack([item[0] for item in batch])
    y = torch.tensor([item[1] for item in batch])
    return X, y

dataloader = DataLoader(
    dataset,
    batch_size=16,
    collate_fn=custom_collate
)
```

## Dataset dengan Transform

```python
import torch
from torch.utils.data import Dataset
import numpy as np

class TransformDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        
        if self.transform:
            x = self.transform(x)
        
        return x, y

# Transform function
def normalize(x):
    return (x - x.mean()) / x.std()

dataset = TransformDataset(
    np.random.randn(100, 10),
    np.random.randint(0, 2, 100),
    transform=normalize
)
```

## Dataset untuk Gambar (dengan torchvision)

```python
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# MNIST dataset
train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Cek satu batch
images, labels = next(iter(train_loader))
print(f"Images shape: {images.shape}")  # torch.Size([64, 1, 28, 28])
print(f"Labels shape: {labels.shape}")  # torch.Size([64])
```

## Iterasi dengan Progress

```python
from torch.utils.data import DataLoader
from tqdm import tqdm

train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

for epoch in range(3):
    for batch_X, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        # Training step
        pass
```

## Weighted Sampling

Untuk dataset yang tidak seimbang:

```python
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

# Dataset tidak seimbang
X = torch.randn(1000, 10)
y = torch.tensor([0]*900 + [1]*100)  # 90% class 0, 10% class 1

dataset = TensorDataset(X, y)

# Hitung weights
class_counts = torch.bincount(y)
class_weights = 1.0 / class_counts.float()
sample_weights = class_weights[y]

# Sampler
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(dataset),
    replacement=True
)

# DataLoader dengan sampler
dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)

# Cek distribusi dalam batch
batch_y = next(iter(dataloader))[1]
print(f"Class distribution in batch: {torch.bincount(batch_y)}")
```

## Multiple Workers

```python
import torch
from torch.utils.data import DataLoader

# Untuk parallel loading
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,      # 4 worker processes
    pin_memory=True,    # Lebih cepat untuk GPU
    persistent_workers=True  # Keep workers alive
)
```

:::{warning}
Di Windows, pastikan kode DataLoader ada di dalam `if __name__ == '__main__':` untuk menghindari error multiprocessing.
:::

## Contoh Lengkap

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

class SimpleDataset(Dataset):
    def __init__(self, size=1000):
        self.X = torch.randn(size, 10)
        self.y = (self.X.sum(dim=1) > 0).long()
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Buat dataset
dataset = SimpleDataset(1000)

# Split
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Simple training loop
model = nn.Linear(10, 2)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(5):
    model.train()
    total_loss = 0
    for X, y in train_loader:
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")
```

## Latihan

1. Buat custom Dataset untuk membaca data dari file CSV
2. Implementasikan data augmentation dalam Dataset
3. Buat DataLoader dengan WeightedRandomSampler
4. Bandingkan kecepatan loading dengan berbagai num_workers
