import numpy as np
from activations import _Activation

# \define net architecture
NET_ARCH = [
   {"input_dim" :  2, "output_dim" : 25, "activation": "relu"},
   {"input_dim" : 25, "output_dim" : 50, "activation": "relu"},
   {"input_dim" : 50, "output_dim" : 25, "activation": "relu"},
   {"input_dim" : 25, "output_dim" :  1, "activation": "sigmoid"},
]

# \random initialize weights and bias 
def init_layers(_Net_arch, _Seed = 99) :
   np.random.seed(_Seed);

   nlayers = len(_Net_arch)
   params = {}
 
   # iteration over net layers
   for I, L in enumerate(_Net_arch):
      layer_idx = I + 1;
      input_size  = L["input_dim"]
      output_size = L["output_dim"]

      params['W' + str(layer_idx)] = np.random.randn(output_size, input_size)*0.1
      params['b' + str(layer_idx)] = np.random.randn(output_size, 1)*0.1

   return params;

# \compute WA + b
def forward_kernel(A, W, b, activation='relu'):
   Z = np.dot(W, A) + b;

   if activation is "relu":
      act_fn = _Activation.relu
   elif activation is "sigmoid":
      act_fn = _Activation.sigmoid
   else:
      raise Exception('Non-supported activation function.')

   return act_fn(Z), Z;

# \whole layers forward computation and intermediate collection
def forward(x, params, net):
   # store info needed for backward step
   _Buf = {}
   x_cur = x

   for idx, layer in enumerate(net):
      layer_idx = idx + 1
      x_pre = x_cur

      W = params["W" + str(layer_idx)]
      b = params["b" + str(layer_idx)]

      x_cur, z = forward_kernel(x_pre, W, b, layer["activation"])

      _Buf["A" + str(idx)] = x_pre
      _Buf["Z" + str(layer_idx)] = z

   return x_cur, _Buf;