import torch as th

x = th.tensor(6.7) # Input 

y = th.tensor(0.0) # Output

w = th.tensor(1.0, requires_grad=True) # Weight

b = th.tensor(0.0, requires_grad=True) # Bias

def binary_cross_entropy_scalar(y_true, y_pred):
    """
    Calculates the Binary Cross-Entropy loss for a single scalar value.

    Returns:
        float: The binary cross-entropy loss.
    """
    # To avoid log(0)
    epsilon = 1e-15
    y_pred = th.clamp(y_pred, epsilon, 1 - epsilon)
    
    loss = -(y_true * th.log(y_pred) + (1 - y_true) * th.log(1 - y_pred))

    return loss


z = w * x + b # Linear transformation

y_pred = th.sigmoid(z) # Sigmoid activation

loss = binary_cross_entropy_scalar(y, y_pred) # Calculate loss

# Backpropagation
loss.backward() # Compute gradients

print(w.grad) # Gradient of loss with respect to weight
print(b.grad) # Gradient of loss with respect to bias