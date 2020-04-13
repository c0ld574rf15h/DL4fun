import numpy as np
from tqdm import tqdm

class NeuralNet():
  ### Initialization ###
  def __init__(self, input_dim, hidden_dim, output_dim, std=1e-4):
    self.params = {}
    self.params['W1'] = std * np.random.randn(input_dim, hidden_dim)
    self.params['b1'] = np.zeros(hidden_dim)
    self.params['W2'] = std * np.random.randn(hidden_dim, output_dim)
    self.params['b2'] = np.zeros(output_dim)

    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.output_dim = output_dim

    self.grads = {}
  
  ### Activation Functions ###
  def sigmoid(self, A):
    """
    Apply sigmoid activation to array
    """
    return 1 / (1 + np.exp(-A))
  
  def softmax(self, A):
    """
    Apply softmax activation to array
    """
    A = np.exp(A)
    # Expand dimension from (x,) to (x, 1)
    sum_of_rows = np.expand_dims(np.sum(A, axis=1), axis=1)
    return A / sum_of_rows

  def num_grad(self, func, X):
    h = 1e-9
    N, M = X.shape
    grad = np.empty_like(X)

    for i in range(N):
      for j in range(M):
        grad = (func(X[i, j]+h) - func(X[i, j]-h)) / 2*h
    return grad
  
  ### Main Dataflow Routines ###
  def loss(self, X, y=None, reg=0):
    N, H, C = X.shape[0], self.hidden_dim, self.output_dim

    # 1. Forward pass
    self.H_ = np.matmul(X, self.params['W1']) + self.params['b1']
    self.H = self.sigmoid(self.H_)
    assert self.H.shape == (N, H), "Hidden layer dimension mismatch"

    self.O = np.matmul(self.H, self.params['W2']) + self.params['b2']
    assert self.O.shape == (N, C), "Output layer dimension mismatch"
    
    O_copy = self.O.copy()
    # Remove numerical instability
    O_copy -= np.expand_dims(np.max(O_copy, axis=1), axis=1)
    self.smx = self.softmax(O_copy)

    if y is None:
      return self.smx
    
    # 2. Compute loss
    loss = np.sum(-np.log(self.smx[np.arange(N), y])) / N
    loss += reg * (np.sum(self.params['W1']**2) + np.sum(self.params['W2']**2))

    # 3. Compute gradients of parameters
    smx_copy = self.smx.copy()
    smx_copy[np.arange(N), y] -= 1
    dO = smx_copy / N

    def f(W2):
      O = np.matmul(self.H, W2) + self.params['b2']
      O_ = O.copy()
      O_ -= np.expand_dims(np.max(O_, axis=1), axis=1)
      smx = self.softmax(O_)
      loss = np.sum(-np.log(smx[np.arange(N), y])) / N
      return loss

    self.grads['W2'] = self.num_grad(f, self.params['W2'])
    # self.grads['W2'] += 2 * reg * self.params['W2']
    # self.grads['b2'] = np.sum(O_copy, axis=0)

    # [N, H] = [N, C] x [C, H]
    # dH = np.matmul(dO, self.params['W2'].T)
    # dH_ = dH * ((1-self.H) * self.H)

    # # [D, H] = [D, N] x [N, H]
    # self.grads['W1'] = np.matmul(X.T, dH_)
    # self.grads['W1'] += 2 * reg * self.params['W1']
    # self.grads['b1'] = np.sum(self.H_, axis=0)

    for param, value in self.params.items():
      if param in self.grads:
        assert value.shape == self.grads[param].shape, \
        'Array <-> Gradient Shape Mismatch ({})'.format(param)

    return loss
    
  def train(self, X, y, epochs, reg=0, lr=0.1):
    """
    Train the neural network

    Input: (X, y, epochs, reg, lr)
      - X (ndarray) : Input data array
      - y (ndarray) : Ground truth labels for input data
      - epochs : Number of epochs to iterate
      - reg : Regularization term for calculating loss
      - lr : Learning rate for updating parameters in the network

    Output : (loss_history)
      - loss_history (list<float>) : List of loss history through training epochs
    """
    loss_history = []

    with tqdm(total=epochs) as pbar:
      for epoch in range(epochs):
        loss_history.append(self.loss(X, y, reg=reg))

        # Update all parameters
        for param in self.params:
          if param in self.grads:
            self.params[param] -= lr * self.grads[param]

        pbar.update(1)
    
    return loss_history
  
  def predict(self, X):
    """
    Predict labels of input data array
    """
    smx_scores = self.loss(X)
    return np.argmax(smx_scores, axis=1)