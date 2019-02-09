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

def output_shape(H, W, nf, stride, pad):
   return int((H-nf+2*pad)/stride + 1), int((W-nf+2*pad)/stride + 1)

def conv_forward(A_prev, W, b, hpars):
   '''
   Implements the forward propagation for a convolution function

   params:
   -- [W] the weights, numpy array of shape (f,f,C_pre,C)
   -- [b] the biases, numpy array of shape (1,1,1,C)
   '''
   m, h_pre, w_pre, c_pre = A_prev.shape
   f, f, c_pre, C = W.shape
   stride = hpars["stride"]
   pad = hpars["pad"]

   H, W_ = output_shape(h_pre, w_pre, f, stride, pad)

   Z = np.zeros((m, H, W_, C))
   A_pre_pad = zero_pad(A_prev, pad)

   for i in range(m):
      a_pre_pad = A_pre_pad[i,:,:,:]
      for h in range(H):
         for w in range(W_):
            for c in range(C): #for each channel, the patch size is fixed
               y0 = stride*h
               y1 = y0+f;
               x0 = stride*w;
               x1 = x0+f;

               a_slice_pre = a_pre_pad[y0:y1,x0:x1,:]
               Z[i,h,w,c] = single_step_conv(a_slice_pre, W[:,:,:,c], b[:,:,:,c])

   assert(Z.shape == (m, H, W_, C))
   cache = (A_prev, W, b, hpars)

   return Z, cache;

A_prev = np.random.randn(10,4,4,3)
W = np.random.randn(2,2,3,8)
b = np.random.randn(1,1,1,8)

hpars = {"pad":2, "stride":2}

Z, conv_cache = conv_forward(A_prev, W, b, hpars)

print("Z = ", Z);