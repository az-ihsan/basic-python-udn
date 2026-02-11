# Training Loop

Training loop adalah inti dari proses melatih neural network. Bagian ini menjelaskan cara menyusun training loop yang baik dan lengkap.

## Komponen Training

1. **Model** - Neural network yang akan dilatih
2. **Loss Function** - Fungsi yang mengukur error
3. **Optimizer** - Algoritma untuk update weights
4. **Data** - DataLoader untuk training dan validation

## Basic Training Loop

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Set seed
torch.manual_seed(42)

# Data
X = torch.randn(1000, 10)
y = (X.sum(dim=1) > 0).long()
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Model
model = nn.Sequential(
    nn.Linear(10, 32),
    nn.ReLU(),
    nn.Linear(32, 2)
)

# Loss dan optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    
    for batch_X, batch_y in dataloader:
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(batch_X)
        
        # Compute loss
        loss = criterion(outputs, batch_y)
        
        # Backward pass
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
```

## Training dengan Validation

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

torch.manual_seed(42)

# Data
X = torch.randn(1000, 10)
y = (X.sum(dim=1) > 0).long()
dataset = TensorDataset(X, y)

# Split
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Model, loss, optimizer
model = nn.Sequential(
    nn.Linear(10, 32),
    nn.ReLU(),
    nn.Linear(32, 2)
)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop dengan validation
num_epochs = 10
for epoch in range(num_epochs):
    # Training phase
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        train_total += y_batch.size(0)
        train_correct += predicted.eq(y_batch).sum().item()
    
    train_acc = 100. * train_correct / train_total
    
    # Validation phase
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += y_batch.size(0)
            val_correct += predicted.eq(y_batch).sum().item()
    
    val_acc = 100. * val_correct / val_total
    
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"  Train Loss: {train_loss/len(train_loader):.4f}, Acc: {train_acc:.2f}%")
    print(f"  Val Loss: {val_loss/len(val_loader):.4f}, Acc: {val_acc:.2f}%")
```

## Loss Functions

### Classification

```python
import torch.nn as nn

# Binary Classification
criterion = nn.BCEWithLogitsLoss()

# Multi-class Classification
criterion = nn.CrossEntropyLoss()

# Multi-class dengan weights
weights = torch.tensor([1.0, 2.0, 1.0])  # Untuk 3 classes
criterion = nn.CrossEntropyLoss(weight=weights)
```

### Regression

```python
import torch.nn as nn

# Mean Squared Error
criterion = nn.MSELoss()

# Mean Absolute Error
criterion = nn.L1Loss()

# Smooth L1 (Huber Loss)
criterion = nn.SmoothL1Loss()
```

## Optimizers

```python
import torch.optim as optim

# SGD
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Adam
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

# AdamW (dengan weight decay yang benar)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

# RMSprop
optimizer = optim.RMSprop(model.parameters(), lr=0.01)
```

## Learning Rate Scheduler

```python
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

# Step decay
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

# Reduce on plateau
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

# Training loop dengan scheduler
for epoch in range(num_epochs):
    train_one_epoch()
    val_loss = validate()
    
    # Untuk StepLR
    scheduler.step()
    
    # Untuk ReduceLROnPlateau
    scheduler.step(val_loss)
    
    print(f"LR: {optimizer.param_groups[0]['lr']}")
```

## Early Stopping

```python
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# Penggunaan
early_stopping = EarlyStopping(patience=5)

for epoch in range(num_epochs):
    train_loss = train_one_epoch()
    val_loss = validate()
    
    early_stopping(val_loss)
    if early_stopping.early_stop:
        print("Early stopping triggered")
        break
```

## Training dengan GPU

```python
import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Model ke GPU
model = model.to(device)

for epoch in range(num_epochs):
    for X_batch, y_batch in train_loader:
        # Data ke GPU
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
```

## Training Function Lengkap

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for X, y in tqdm(dataloader, desc="Training"):
        X, y = X.to(device), y.to(device)
        
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += y.size(0)
        correct += predicted.eq(y).sum().item()
    
    return total_loss / len(dataloader), 100. * correct / total

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for X, y in tqdm(dataloader, desc="Validating"):
            X, y = X.to(device), y.to(device)
            
            outputs = model(X)
            loss = criterion(outputs, y)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
    
    return total_loss / len(dataloader), 100. * correct / total

def train(model, train_loader, val_loader, criterion, optimizer, 
          scheduler, device, num_epochs):
    best_val_acc = 0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
        
        # Scheduler step
        if scheduler:
            scheduler.step(val_loss)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"Saved best model with val_acc: {val_acc:.2f}%")
    
    return model
```

## Gradient Clipping

```python
# Clip by norm
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Clip by value
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)

# Dalam training loop
for X, y in train_loader:
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    
    # Clip gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
```

## Latihan

1. Implementasikan training loop dengan logging ke TensorBoard
2. Tambahkan early stopping dan model checkpointing
3. Implementasikan learning rate finder
4. Buat training loop untuk regression task
