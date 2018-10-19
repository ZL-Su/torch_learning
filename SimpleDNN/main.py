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

# \cost determined by the binary crossentropy
def cost(samples, predictions):
   nsamples = samples.shape[1]
   cost_val = -1/m * (np.dot(predictions, np.log(samples).T) + np.dot(1-predictions, np.log(1-predictions).T))

   return np.squeeze(cost_val);

def decision(probs):
   _Probs = np.copy(probs)
   _Probs[_Probs > 0.5] = 1
   _Probs[_Probs <= 0.5] = 0

   return _Probs;

# \calc. accuracy
def accuracy(samples, predictions):
   _Samples = decision(samples);

   return (_Samples == predictions).all(axis=0).mean()

def backward_kernel(dA, W, b, Z, A, activation="relu"):
   # number of examples
   m = A.shape[1]

   if activation is "relu":
      act_fn = _Activation.relu_backward
   elif activation is "sigmoid":
      act_fn = _Activation.sigmoid_backward
   else:
      raise Exception('Non-supported activation function.')

   # calc. of the activation function derivative
   dZ = act_fn(dA, Z)

   # derivative of the matrix W
   dW = np.dot(dZ, A.T)/m

   # derivative of the vector b
   db = np.sum(dZ, axis=1, keepdims=True)/m

   # derivative of the matrix A
   dA_pre = np.dot(W.T, dZ)

   return dA_pre, dW, db

def backward(sampl, predi, buf, params, net):
   grads = {}

   m = predi.shape[1]
   predi = predi.reshape(sampl.shape)

   # initiation of gradient descent algorithm
   dA_pre = -(np.divide(predi, sampl) - np.divide(1-predi, 1-sampl))

   for idx_pre, layer in reversed(list(enumerate(net))):
      idx_cur = idx_pre + 1

      activ_fn_cur = layer["activation"]

      dA_cur = dA_pre

      A_pre = buf["A" + str(idx_pre)]
      Z_cur = buf["Z" + str(idx_cur)]

      W_cur = params["W" + str(idx_cur)]
      b_cur = params["b" + str(idx_cur)]

      dA_pre, dW_cur, db_cur = backward_kernel(dA_cur, W_cur, b_cur, Z_cur, A_pre, activ_fn_cur)

      grads["dW" + str(idx_cur)] = dW_cur
      grads["db" + str(idx_cur)] = db_cur

   return grads;
