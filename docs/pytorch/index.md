# PyTorch

PyTorch adalah framework open-source machine learning yang mempercepat jalur dari prototyping penelitian hingga deployment produksi. PyTorch menawarkan komputasi tensor dengan akselerasi GPU yang kuat dan deep neural networks yang dibangun di atas sistem autograd berbasis tape.

## Mengapa PyTorch?

- **Dynamic Computation Graph** - Fleksibel untuk debugging dan eksperimen
- **Pythonic** - Terasa natural bagi pengguna Python
- **GPU Acceleration** - Mudah berpindah antara CPU dan GPU
- **Ekosistem Kaya** - TorchVision, TorchText, TorchAudio, dll

## Instalasi

```bash
# CPU only
pip install torch

# Dengan CUDA (GPU support)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Import PyTorch

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Set seed untuk reprodusibilitas
torch.manual_seed(42)
```

## Daftar Materi

```{toctree}
:maxdepth: 1

tensor
autograd
nn_module
dataloader
training_loop
saving_loading
```

## Contoh Cepat

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 1. Buat data sederhana
X = torch.randn(100, 10)
y = torch.randn(100, 1)

# 2. Definisikan model
model = nn.Sequential(
    nn.Linear(10, 32),
    nn.ReLU(),
    nn.Linear(32, 1)
)

# 3. Loss dan optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 4. Training loop
for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
```

## PyTorch vs NumPy

```python
import torch
import numpy as np

# NumPy
np_arr = np.array([1, 2, 3])
print(np_arr * 2)  # [2 4 6]

# PyTorch
torch_tensor = torch.tensor([1, 2, 3])
print(torch_tensor * 2)  # tensor([2, 4, 6])

# Konversi
tensor_from_np = torch.from_numpy(np_arr)
np_from_tensor = torch_tensor.numpy()
```

## GPU Support

```python
import torch

# Cek apakah CUDA tersedia
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"Device name: {torch.cuda.get_device_name(0)}")

# Pindahkan tensor ke GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tensor = torch.randn(3, 3).to(device)
print(f"Tensor device: {tensor.device}")
```

## Alur Kerja PyTorch

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Data      │ → │  DataLoader │ → │   Model     │
│   (Tensor)  │    │   (Batch)   │    │ (nn.Module) │
└─────────────┘    └─────────────┘    └──────┬──────┘
                                              │
        ┌─────────────────────────────────────┘
        │
        ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Forward   │ → │    Loss     │ → │  Backward   │
│   Pass      │    │   Function  │    │   Pass      │
└─────────────┘    └─────────────┘    └──────┬──────┘
                                              │
                                              ▼
                                      ┌─────────────┐
                                      │  Optimizer  │
                                      │   Step      │
                                      └─────────────┘
```

## Langkah Selanjutnya

Lanjutkan ke [Tensor](tensor.md) untuk mempelajari struktur data dasar PyTorch.
