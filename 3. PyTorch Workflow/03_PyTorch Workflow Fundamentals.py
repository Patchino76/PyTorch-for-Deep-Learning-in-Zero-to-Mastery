#%%
# 01. PyTorch Workflow Fundamentals

import torch
from torch import nn # nn contains all of PyTorch's building blocks for neural networks
import matplotlib.pyplot as plt

# Check PyTorch version
torch.__version__
# %%
# Create *known* parameters
weight = 0.7
bias = 0.3

# Create data
start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias

X[:10], y[:10], len(X)
# %%
# Create train/test split
train_split = int(0.8 * len(X)) # 80% of data used for training set, 20% for testing 
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

len(X_train), len(y_train), len(X_test), len(y_test)
# %%
#PLOT data
def plot_predictions(train_data=X_train, 
                    train_labels=y_train,
                    test_data=X_test, 
                    test_labels=y_test, 
                    predictions=None):
    plt.figure(figsize=(10,7))
    #plot training data
    plt.scatter(train_data, train_labels, c='b', s=4, label='train data')
    #plot test data
    plt.scatter(test_data, test_labels, c='g', s=4, label='test data')

    if predictions is not None:
        plt.scatter(test_data, predictions, c='r', s=4, label='predictions')

    plt.legend(prop={'size':14})

plot_predictions()
# %%
## Create a Linear Regression model class

class LinearRegressionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        #Initialize model parameters
        self.weights = nn.Parameter(torch.randn(1,requires_grad=True, dtype=float))
        self.bias = nn.Parameter(torch.randn(1,requires_grad=True, dtype=float))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias


# %%
# torch create a random seed
torch.manual_seed(42) 

#create an istance of the model LinearRegressionModel

#check parameters
model_0 = LinearRegressionModel()
model_0.parameters() # returns a generator
list(model_0.parameters())

# %%
#List named parameters
model_0.state_dict()
# %%
#Making predictions with torch.inference_mode()
with torch.inference_mode():
    y_preds = model_0(X_test)

y_preds
# %%
plot_predictions(predictions=y_preds)


# %%
#Setup a loss func with torch
loss_fn = nn.L1Loss()

#setup an optimizer
optimizer = torch.optim.SGD(model_0.parameters(), lr=0.001)


# %%
torch.manual_seed(42)
#Buld training loop
epochs = 10000

loss_arr = []

for epoch in range(epochs):
    #set the model in training mode
    model_0.train() #set the parameters that require grad to true

    #1. Forward pass
    y_pred = model_0(X_train)

    #2. calc loss
    loss = loss_fn(y_pred, y_train)
    loss_arr.append(loss.detach().numpy())    
    #3. calc grad
    optimizer.zero_grad()

    #4. backward prop on the loss with respects to the parameters
    loss.backward()

    #5. step the optimizer -> perform gradient descent
    optimizer.step()


    model_0.eval() #turns off gradient tracking

    # Testing loop...
    with torch.inference_mode():
        test_pred = model_0(X_test)
        test_loss = loss_fn(test_pred, y_test)

    if epoch %10 == 0:
        print(f'Epoch: {epoch} Train Loss: {loss} Test Loss: {test_loss}')





# %%
plot_predictions(predictions=test_pred)
plt.show()
plt.plot(loss_arr)
plt.show()
# %%
#Saving the model
# torch.save()
# torch.load()
# torch.nn.load_state_dict()

from pathlib import Path
MODEL_PATH = Path("C:/MFC_Scripts\COURSES/Udemy - PyTorch for Deep Learning in 2023 Zero to Mastery 2022-11")
MODEL_NAME = "01_workflow_model_0.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

torch.save(model_0.state_dict(), MODEL_SAVE_PATH)
print(f"Saving model to {MODEL_SAVE_PATH}")


# %%
# Loading model
model_1 = LinearRegressionModel()
# model_1 = torch.load(MODEL_SAVE_PATH)
model_1.load_state_dict(torch.load(MODEL_SAVE_PATH))


