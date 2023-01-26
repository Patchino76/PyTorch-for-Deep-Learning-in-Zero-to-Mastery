#%%
import torch
from torch import nn # nn contains all of PyTorch's building blocks for neural networks
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np 


# %% Multivariate sampling
X = torch.linspace(0, 1, 1000).unsqueeze(dim=1) # X values 
noise = torch.randn_like(X) # create noise 
y = 5*X + noise

split = int(0.8*len(X))
x_train, y_train = X[:split], y[:split]
x_test, y_test = X[split:], y[split:]



# %%
class LinearRegressionModelV1(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        #Initialize model parameters
        self.weights = nn.Parameter(torch.randn(1,requires_grad=True, dtype=float))
        self.bias = nn.Parameter(torch.randn(1,requires_grad=True, dtype=float))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias


class LinearRegressionModelV2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features=1, out_features=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)
# %%

model = LinearRegressionModelV2()
print(model.state_dict())

loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(params=model.parameters(),  lr=0.01)

# %%
#Train model
train_losses = []
test_losses = []
epochs = 10000

for epoch in range(epochs):
    model.train()
    y_train_hat = model(x_train)
    train_loss = loss_fn(y_train_hat, y_train)
    train_losses.append(train_loss.detach().numpy())

    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    model.eval()
    with torch.inference_mode():
        y_test_hat = model(x_test)
        test_loss = loss_fn(y_test_hat, y_test)
        test_losses.append(test_loss.detach().numpy())

    if epoch %10 == 0:
        print(f'Epoch: {epoch} Train Loss: {train_loss} Test Loss: {test_loss}')




# %%
#PLOT data
def plot_predictions(train_data=x_train, 
                    train_labels=y_train,
                    test_data=x_test, 
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



# %%
plot_predictions(predictions=y_test_hat)
plt.show()

# %%
plt.plot(train_losses,  label='train loss')
plt.plot(test_losses,  label='test loss')
plt.legend(prop={'size':14})

# %%
plt.plot(y_train)
plt.plot(y_train_hat.squeeze().detach().numpy())
plt.show()
# %%
