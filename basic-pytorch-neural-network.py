import torch
from torch import nn, tensor # nn contains all of PyTorchs building blocks for neural networks

'''
What our model does:
1, Start with randomly initialized parameters (weights and bias)
2. Look at the data and adjust the parameters to better fit the data 
(this is called training) making them closer to the true parameters 
(weight = 0.7, bias = 0.3) that we used to create the data

How does it do so?
1. Gradient descent (an optimization algorithm to minimize the loss function)
2. Backpropagation via autograd (automatic differentiation)
'''

# Create known parameters
weight = 0.7
bias = 0.3
# Values of weight and bias that we will use to create our data, 
# and that our model will try to reach through training

# Create data
start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias

print(X[:2], y[:2]) # Show the first 2 samples of X and y

# Create train/test split
train_split = int(0.8 * len(X)) # 80% of data used for training set, 20% for testing 
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

print(len(X_train), len(y_train), len(X_test), len(y_test))

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1, 
                                                requires_grad=True, 
                                                dtype=torch.float32))
        self.bias = nn.Parameter(torch.randn(1, 
                                             requires_grad=True, 
                                             dtype=torch.float32))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
            return self.weights * x + self.bias # Return the predicted values given the inputs and parameters of the model
    
loss_fn = nn.L1Loss() # Create a loss function (L1 loss, also known as mean absolute error)

torch.manual_seed(42) # Set the random seed for reproducibility (this ensures that the randomly initialized parameters of the model are the same each time we run the code)
model = LinearRegressionModel()

# Parameter is a value that the model can update to make better predictions after looking at the data (e.g., weights and bias)

optimizer = torch.optim.SGD(params=model.parameters(),  # Create an optimizer (stochastic gradient descent) to update the parameters of the model during training
                            lr=0.01) # Set the learning rate (how much the parameters should be updated during each step of training)

epochs = 100
        
print(list(model.parameters())) # Show the randomly initialized parameters of the model (weights and bias) before training
print(model.state_dict()) # Show the state of the model's parameters (weights and bias) before training
 
print(X_test) # Show the test set inputs before training

# Make a prediction before training (this will be random because the model's parameters are randomly initialized)
with torch.inference_mode(): # Inference mode is a context manager that disables gradient calculation, which can save memory and speed up computations when you are only doing inference (i.e., making predictions) and not training.
    y_pred = model(X_test)
print(y_pred) # Showing the predicted values for the test set before training

# Train the model

for epoch in range(epochs):
    model.train() # Set the model to training mode (this is the default mode, but it's good practice to explicitly set it)
    
    y_pred = model(X_train) # Make a prediction using the model on the training data
    
    loss = loss_fn(y_pred, y_train) # Calculate the loss (how far off the predictions are from the true values)
    
    optimizer.zero_grad() # Zero the gradients (this is important because by default, gradients are accumulated in PyTorch)
    
    loss.backward() # Backpropagation (calculate the gradients of the loss with respect to the parameters of the model)
    
    optimizer.step() # Update the parameters of the model using the calculated gradients

    # Testing the model after each epoch of training
    model.eval() # Set the model to evaluation mode
    with torch.inference_mode(): # Inference mode  disables gradient calculation, which can save memory and speed up computations

        # Forward pass: compute predicted y by passing x to the model
        test_pred = model(X_test) # Make a prediction using the model on the test data

        # Compute and print loss
        test_loss = loss_fn(test_pred, y_test) # Calculate the loss on the test set
        print(f"Test Loss: {test_loss.item()}") # Print the test loss

    print(f"Epoch: {epoch} | Loss: {loss.item()} | Test Loss: {test_loss.item()}") # Print the epoch number and the loss for that epoch
    print(model.state_dict()) # Show the state of the model's parameters (weights and bias) after each epoch of training

