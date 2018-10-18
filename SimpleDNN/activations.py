import numpy as np

class _Activation:
   def sigmoid(z) :
      return 1/(1+np.exp(-z))

   def relu(z) :
      return np.maximum(0, z)

   def sigmoid_backward(grad, z):
      _Val = sigmoid(z)
      return grad*_Val*(1-_Val)

   def relu_backward(grad, z):
      dZ = np.array(grad, copy=True)
      dZ[z < 0] = 0;
      return dZ;