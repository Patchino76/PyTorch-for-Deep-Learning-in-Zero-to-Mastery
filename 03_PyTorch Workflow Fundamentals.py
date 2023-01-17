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
#