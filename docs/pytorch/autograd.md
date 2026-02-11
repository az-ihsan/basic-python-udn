# Autograd

Autograd adalah sistem diferensiasi otomatis di PyTorch. Sistem ini melacak operasi pada tensor dan secara otomatis menghitung gradien untuk backpropagation.

## Konsep Dasar

```python
import torch

# Buat tensor dengan requires_grad=True
x = torch.tensor([2.0, 3.0], requires_grad=True)
print(f"x: {x}")
print(f"requires_grad: {x.requires_grad}")
```

## Computational Graph

```python
import torch

# Forward pass membangun computational graph
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2
z = 2 * y + 3

print(f"x: {x}")
print(f"y = x^2: {y}")
print(f"z = 2y + 3: {z}")

# Backward pass menghitung gradien
z.backward()

# dz/dx = dz/dy * dy/dx = 2 * 2x = 4x = 8
print(f"dz/dx: {x.grad}")  # tensor([8.])
```

## Gradient Computation

### Scalar Output

```python
import torch

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x.sum()  # Scalar output

y.backward()
print(f"Gradient: {x.grad}")  # tensor([1., 1., 1.])
```

### Vector Output

```python
import torch

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x ** 2  # Vector output

# Perlu gradient vector untuk non-scalar output
y.backward(torch.ones_like(y))
print(f"Gradient: {x.grad}")  # tensor([2., 4., 6.])
```

### Dengan Loss Function

```python
import torch
import torch.nn.functional as F

# Simulasi output model dan target
predictions = torch.tensor([0.1, 0.9, 0.2], requires_grad=True)
targets = torch.tensor([0.0, 1.0, 0.0])

# Hitung loss
loss = F.mse_loss(predictions, targets)
print(f"Loss: {loss.item():.4f}")

# Backward
loss.backward()

# Gradien predictions
print(f"Gradients: {predictions.grad}")
```

## Detaching dan No Grad

### detach()

```python
import torch

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x ** 2

# Detach dari graph
z = y.detach()
print(f"z requires_grad: {z.requires_grad}")  # False

# z tidak terhubung ke x
```

### torch.no_grad()

```python
import torch

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

# Dalam context no_grad, operasi tidak dilacak
with torch.no_grad():
    y = x ** 2
    print(f"y requires_grad: {y.requires_grad}")  # False
```

:::{tip}
Gunakan `torch.no_grad()` saat inference untuk menghemat memori dan mempercepat komputasi.
:::

## Gradient Accumulation

```python
import torch

x = torch.tensor([1.0], requires_grad=True)

# Gradien terakumulasi
y1 = x ** 2
y1.backward()
print(f"After first backward: {x.grad}")  # tensor([2.])

y2 = x ** 3
y2.backward()
print(f"After second backward: {x.grad}")  # tensor([5.]) = 2 + 3

# Untuk training, perlu zero gradients
x.grad.zero_()
y3 = x ** 2
y3.backward()
print(f"After zero and backward: {x.grad}")  # tensor([2.])
```

## Contoh dengan Neural Network

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# Set seed
torch.manual_seed(42)

# Simple network
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(4, 3)
        self.linear2 = nn.Linear(3, 1)
    
    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x

net = SimpleNet()

# Input
x = torch.randn(4)
target = torch.tensor([1.0])

# Forward
output = net(x)
loss = torch.abs(output - target)

# Backward
loss.backward()

# Lihat gradien
for name, param in net.named_parameters():
    if param.grad is not None:
        print(f"{name} grad shape: {param.grad.shape}")
```

## retain_graph

```python
import torch

x = torch.tensor([1.0], requires_grad=True)
y = x ** 2
z = y ** 2

# Backward pertama
z.backward(retain_graph=True)
print(f"First backward: {x.grad}")

# Reset grad
x.grad.zero_()

# Backward kedua (graph masih ada)
z.backward()
print(f"Second backward: {x.grad}")
```

## Higher-Order Gradients

```python
import torch

x = torch.tensor([2.0], requires_grad=True)

# First derivative
y = x ** 3
grad1 = torch.autograd.grad(y, x, create_graph=True)[0]
print(f"dy/dx = 3x^2: {grad1}")  # tensor([12.])

# Second derivative
grad2 = torch.autograd.grad(grad1, x)[0]
print(f"d2y/dx2 = 6x: {grad2}")  # tensor([12.])
```

## Gradient Clipping

```python
import torch
import torch.nn as nn

# Model sederhana
model = nn.Linear(10, 5)
x = torch.randn(3, 10)
y = model(x)
loss = y.sum()
loss.backward()

# Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Atau clip by value
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
```

## Custom Autograd Function

```python
import torch

class MyReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input

# Penggunaan
x = torch.randn(3, requires_grad=True)
y = MyReLU.apply(x)
y.sum().backward()
print(f"Input: {x}")
print(f"Grad: {x.grad}")
```

## Debugging Autograd

```python
import torch

# Enable anomaly detection
torch.autograd.set_detect_anomaly(True)

x = torch.tensor([1.0], requires_grad=True)
y = x ** 2

# Ini akan memberikan traceback yang lebih informatif jika ada error
try:
    y.backward()
except Exception as e:
    print(f"Error: {e}")
```

## Tips Autograd

1. **Selalu zero_grad()** sebelum backward() dalam training loop
2. **Gunakan no_grad()** untuk inference
3. **Detach** tensor yang tidak perlu gradien
4. **Perhatikan in-place operations** yang dapat merusak graph

## Latihan

1. Hitung gradien dari fungsi kompleks secara manual dan verifikasi dengan autograd
2. Implementasikan custom autograd function untuk sigmoid
3. Debug neural network yang menghasilkan NaN gradient
4. Implementasikan gradient penalty untuk regularization
