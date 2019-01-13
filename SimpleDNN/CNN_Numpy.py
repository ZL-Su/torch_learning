import numpy as np
import h5py
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (5.0, 5.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'jet'

np.random.seed(1)

def zero_pad(X, size):
   _Padded = np.pad(X, ((0,0),(size,size),(size,size),(0,0)),'constant')
   return _Padded;

x = np.random.randn(4,3,3,2)
x_pad = zero_pad(x, 2)
fig,axarr = plt.subplots(1,2)
axarr[0].set_title('x')
axarr[0].imshow(x[0,:,:,0])
axarr[1].set_title('y')
axarr[1].imshow(x_pad[0,:,:,0])

def single_step_conv(a_slice_prev, W, b):
   s = a_slice_prev*W
   Z = np.sum(s)
   Z = Z+b
   return Z;

a_slice_prev = np.random.randn(4,4,3)
W = np.random.randn(4,4,3)
b = np.random.randn(1,1,1)
print(a_slice_prev)
print(W)
print(b)

Z = single_step_conv(a_slice_prev, W, b)

print("Z = ", Z);