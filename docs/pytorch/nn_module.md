# Neural Network Module

`torch.nn.Module` adalah base class untuk semua neural network di PyTorch. Modul ini menyediakan cara terstruktur untuk membangun dan mengorganisir model.

## Mendefinisikan Model

### Dengan nn.Sequential

```python
import torch
import torch.nn as nn

# Sequential: urutan layer linear
model = nn.Sequential(
    nn.Linear(10, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 1)
)

# Forward pass
x = torch.randn(5, 10)  # batch_size=5, features=10
output = model(x)
print(f"Output shape: {output.shape}")  # torch.Size([5, 1])
```

### Dengan Custom Class

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(4, 3)
        self.linear2 = nn.Linear(3, 1)
    
    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x

net = Net()
print(net)

# Forward pass
x = torch.randn(4)
output = net(x)
print(f"Output: {output}")
```

## Layer Umum

### Linear Layer

```python
import torch.nn as nn

# Fully connected layer
linear = nn.Linear(in_features=10, out_features=5)

# Dengan bias=False
linear_no_bias = nn.Linear(10, 5, bias=False)
```

### Activation Functions

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

x = torch.randn(3)

# Sebagai module
relu = nn.ReLU()
sigmoid = nn.Sigmoid()
tanh = nn.Tanh()
softmax = nn.Softmax(dim=-1)

# Sebagai functional
y = F.relu(x)
y = F.sigmoid(x)
y = F.tanh(x)
y = F.softmax(x, dim=-1)

# Variasi ReLU
leaky_relu = nn.LeakyReLU(negative_slope=0.01)
elu = nn.ELU(alpha=1.0)
gelu = nn.GELU()
```

### Convolutional Layers

```python
import torch.nn as nn

# 2D Convolution
conv2d = nn.Conv2d(
    in_channels=3,       # RGB
    out_channels=16,
    kernel_size=3,
    stride=1,
    padding=1
)

# 1D Convolution
conv1d = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3)
```

### Pooling Layers

```python
import torch.nn as nn

# Max pooling
maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

# Average pooling
avgpool = nn.AvgPool2d(kernel_size=2, stride=2)

# Adaptive pooling (output size tetap)
adaptive_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
```

### Normalization Layers

```python
import torch.nn as nn

# Batch Normalization
bn = nn.BatchNorm2d(num_features=64)

# Layer Normalization
ln = nn.LayerNorm(normalized_shape=64)

# Instance Normalization
in_norm = nn.InstanceNorm2d(num_features=64)
```

### Dropout

```python
import torch.nn as nn

dropout = nn.Dropout(p=0.5)

# Dalam model
class NetWithDropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 50)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(50, 10)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
```

### Embedding

```python
import torch
import torch.nn as nn

# Embedding layer
embedding = nn.Embedding(num_embeddings=1000, embedding_dim=64)

# Lookup
indices = torch.tensor([1, 5, 10])
embedded = embedding(indices)
print(f"Embedded shape: {embedded.shape}")  # torch.Size([3, 64])
```

### RNN, LSTM, GRU

```python
import torch.nn as nn

# LSTM
lstm = nn.LSTM(
    input_size=10,
    hidden_size=20,
    num_layers=2,
    batch_first=True,
    dropout=0.1,
    bidirectional=True
)

# GRU
gru = nn.GRU(input_size=10, hidden_size=20, num_layers=2, batch_first=True)
```

## Contoh Model

### MLP Classifier

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

model = MLPClassifier(input_dim=784, hidden_dim=256, num_classes=10)
print(model)
```

### CNN untuk Gambar

```python
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # Input: (batch, 3, 32, 32)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # (batch, 32, 16, 16)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # (batch, 64, 8, 8)
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
```

## Parameter dan Akses

```python
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 5)
)

# Lihat semua parameters
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")

# Total parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total: {total_params}, Trainable: {trainable_params}")

# Akses layer spesifik
print(model[0].weight.shape)  # torch.Size([20, 10])
```

## Train dan Eval Mode

```python
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(10, 20),
    nn.Dropout(0.5),
    nn.Linear(20, 5)
)

# Training mode (dropout aktif)
model.train()
print(f"Training: {model.training}")

# Evaluation mode (dropout non-aktif)
model.eval()
print(f"Training: {model.training}")
```

:::{important}
Selalu panggil `model.eval()` sebelum inference dan `model.train()` sebelum training.
:::

## Move ke Device

```python
import torch
import torch.nn as nn

model = nn.Linear(10, 5)

# Pindah ke GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Input juga harus di device yang sama
x = torch.randn(3, 10).to(device)
output = model(x)
```

## Latihan

1. Buat model MLP dengan 3 hidden layers dan dropout
2. Implementasikan CNN sederhana untuk MNIST
3. Buat model dengan residual connection
4. Hitung jumlah parameter trainable dalam model
