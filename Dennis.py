
import numpy as np
loaded_data = np.load('arrays.npz')
lx = loaded_data['lx'] #(20, 256)
ux = loaded_data['ux'] #(20, 256)
W_Q=loaded_data['W_Q'] #(4, 64, 256)   4: attention head number  64:output dim for an attention head, 256:input dim
W_K=loaded_data['W_K'] #(4, 64, 256)


print(lx.shape)
print(ux.shape)
print(W_Q.shape)
print(W_K.shape)

W_q=W_Q[0,:,:]
W_k=W_K[0,:,:]  # get the matrix for the first attention head

W_q=W_q.T()  # this Transformation can make the W_q be same with what in the paper
W_k=W_k.T()

