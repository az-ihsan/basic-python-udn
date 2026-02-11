# Saving dan Loading

Menyimpan dan memuat model adalah bagian penting dari workflow machine learning. PyTorch menyediakan beberapa cara untuk melakukan ini.

## Menyimpan State Dict (Disarankan)

```python
import torch
import torch.nn as nn

# Model
model = nn.Sequential(
    nn.Linear(10, 32),
    nn.ReLU(),
    nn.Linear(32, 2)
)

# Simpan state dict
torch.save(model.state_dict(), 'model_weights.pth')

# Load state dict
model_loaded = nn.Sequential(
    nn.Linear(10, 32),
    nn.ReLU(),
    nn.Linear(32, 2)
)
model_loaded.load_state_dict(torch.load('model_weights.pth'))
model_loaded.eval()  # Set ke evaluation mode
```

## Menyimpan Seluruh Model

```python
import torch
import torch.nn as nn

model = nn.Linear(10, 5)

# Simpan seluruh model
torch.save(model, 'full_model.pth')

# Load seluruh model
model_loaded = torch.load('full_model.pth')
model_loaded.eval()
```

:::{warning}
Menyimpan seluruh model menggunakan pickle dan bergantung pada struktur class yang tersedia saat loading. Disarankan menggunakan state_dict untuk portabilitas lebih baik.
:::

## Checkpoint untuk Training

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Komponen
model = nn.Linear(10, 5)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10)
epoch = 50
loss = 0.5

# Simpan checkpoint
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'loss': loss,
}
torch.save(checkpoint, 'checkpoint.pth')

# Load checkpoint
checkpoint = torch.load('checkpoint.pth')

model_loaded = nn.Linear(10, 5)
optimizer_loaded = optim.Adam(model_loaded.parameters(), lr=0.001)
scheduler_loaded = optim.lr_scheduler.StepLR(optimizer_loaded, step_size=10)

model_loaded.load_state_dict(checkpoint['model_state_dict'])
optimizer_loaded.load_state_dict(checkpoint['optimizer_state_dict'])
scheduler_loaded.load_state_dict(checkpoint['scheduler_state_dict'])
epoch_loaded = checkpoint['epoch']
loss_loaded = checkpoint['loss']

print(f"Resumed from epoch {epoch_loaded}, loss {loss_loaded}")
```

## Save Best Model

```python
import torch

class ModelCheckpoint:
    def __init__(self, filepath, monitor='val_loss', mode='min'):
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.best = float('inf') if mode == 'min' else float('-inf')
    
    def __call__(self, current_value, model):
        if self.mode == 'min':
            is_best = current_value < self.best
        else:
            is_best = current_value > self.best
        
        if is_best:
            self.best = current_value
            torch.save(model.state_dict(), self.filepath)
            print(f"Saved best model with {self.monitor}: {current_value:.4f}")
            return True
        return False

# Penggunaan
checkpoint = ModelCheckpoint('best_model.pth', monitor='val_acc', mode='max')

for epoch in range(num_epochs):
    val_acc = train_and_validate()
    checkpoint(val_acc, model)
```

## Saving untuk Inference

```python
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 32)
        self.fc2 = nn.Linear(32, 2)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

model = MyModel()
model.eval()

# Simpan untuk inference
torch.save({
    'model_state_dict': model.state_dict(),
    'model_config': {'input_dim': 10, 'hidden_dim': 32, 'output_dim': 2}
}, 'inference_model.pth')

# Load untuk inference
checkpoint = torch.load('inference_model.pth')
config = checkpoint['model_config']

model = MyModel()  # Buat ulang dengan config jika perlu
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Inference
with torch.no_grad():
    x = torch.randn(5, 10)
    output = model(x)
    print(output)
```

## Menyimpan di Device Berbeda

### Save dari GPU, Load ke CPU

```python
import torch

# Simpan model yang ada di GPU
torch.save(model.state_dict(), 'model.pth')

# Load ke CPU
device = torch.device('cpu')
model.load_state_dict(torch.load('model.pth', map_location=device))
```

### Load ke GPU Tertentu

```python
import torch

# Load ke GPU 0
model.load_state_dict(torch.load('model.pth', map_location='cuda:0'))

# Atau gunakan device
device = torch.device('cuda:0')
model.load_state_dict(torch.load('model.pth', map_location=device))
model.to(device)
```

## TorchScript untuk Production

```python
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)
    
    def forward(self, x):
        return torch.relu(self.fc(x))

model = MyModel()
model.eval()

# Trace model
example_input = torch.randn(1, 10)
traced_model = torch.jit.trace(model, example_input)

# Simpan
traced_model.save('model_traced.pt')

# Load (tidak perlu definisi class)
loaded = torch.jit.load('model_traced.pt')
output = loaded(example_input)
print(output)
```

## ONNX Export

```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(10, 32),
    nn.ReLU(),
    nn.Linear(32, 2)
)
model.eval()

# Dummy input
dummy_input = torch.randn(1, 10)

# Export ke ONNX
torch.onnx.export(
    model,
    dummy_input,
    'model.onnx',
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)
```

## Training Loop dengan Checkpoint

```python
import torch
import torch.nn as nn
import torch.optim as optim
import os

def train_with_checkpoint(model, train_loader, val_loader, criterion,
                          optimizer, scheduler, num_epochs, checkpoint_dir='checkpoints'):
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    start_epoch = 0
    best_val_loss = float('inf')
    
    # Resume dari checkpoint jika ada
    checkpoint_path = os.path.join(checkpoint_dir, 'latest.pth')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        print(f"Resumed from epoch {start_epoch}")
    
    for epoch in range(start_epoch, num_epochs):
        # Training
        model.train()
        for X, y in train_loader:
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X, y in val_loader:
                outputs = model(X)
                val_loss += criterion(outputs, y).item()
        val_loss /= len(val_loader)
        
        scheduler.step(val_loss)
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
        }
        torch.save(checkpoint, os.path.join(checkpoint_dir, 'latest.pth'))
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))
        
        print(f"Epoch {epoch+1}/{num_epochs}, Val Loss: {val_loss:.4f}")
    
    return model
```

## Latihan

1. Implementasikan checkpoint yang menyimpan top-3 model terbaik
2. Buat fungsi untuk mengekspor model ke ONNX dan memvalidasinya
3. Implementasikan resume training dari checkpoint
4. Buat script untuk konversi model antar device (CPU/GPU)
