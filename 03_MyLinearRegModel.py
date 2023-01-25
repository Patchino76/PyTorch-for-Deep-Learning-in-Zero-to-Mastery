#%%
import torch
from torch import nn # nn contains all of PyTorch's building blocks for neural networks
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np 


# %% Multivariate sampling
mean = torch.tensor([0.0, 0.0])
covar = torch.tensor([[0.8, 0.7], [0.7, 0.8]])
slope = covar[0][1]/covar[0][0] 
# Create a multivariate normal distribution object
mvn = torch.distributions.MultivariateNormal(mean, covar)

# Sample from the multivariate normal distribution
sample = mvn.sample((1000,))
X = sample[:,0].unsqueeze(dim=1)
y = sample[:,1].unsqueeze(dim=1)

split = int(0.8*len(X))
x_train, y_train = X[:split], y[:split]
x_test, y_test = X[split:], y[split:]

xax = torch.arange(-3.0, 3.0, 0.01)
yax = xax * slope
plt.scatter(X, y)
plt.plot(xax, yax)

# %%
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

epochs = 100

for epoch in range(epochs):
    model.train()
    y_train_hat = model(x_train.unsqueeze(dim=1))
    train_loss = loss_fn(y_train_hat, y_train)
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    model.eval()
    with torch.inference_mode():
        y_test_hat = model(x_test)
        test_loss = loss_fn(y_test_hat, y_test)

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
