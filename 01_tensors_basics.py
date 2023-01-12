import torch
import numpy as np
#%%

tens = torch.tensor([[[1,2,3],
                      [4,5,6],
                      [7,8,9]]])

tens.shape

#%%
tens[0,:,0]

#%%
tens[0,:,2]

#%%

arr = np.arange(1.0, 8.0)
ts = torch.from_numpy(arr)

print(ts)


ts = ts.type(torch.float32)
print(ts.dtype)


#%%
tens = torch.ones(10, dtype=torch.float16)
arr = tens.numpy()
tens, arr