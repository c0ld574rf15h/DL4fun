import numpy as np
from tqdm import trange

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
  
  ### Activation Functions ###
  def sigmoid(self, A):
    """
    Apply sigmoid activation to array
    """
    return 1 / (1 + np.exp(-A))
  
  def relu(self, A):
    """
    Apply ReLU activation to array
    """
    return np.maximum(0, A)
  
  def softmax(self, A):
    """
    Apply softmax activation to array
    """
    A_exp = np.exp(A)
    sum_of_rows = np.sum(A_exp, axis=1, keepdims=True)

    return A_exp / sum_of_rows
  
  ### Main Dataflow Routines ###
  def loss(self, X, y=None, act='relu', reg=0):
    N, H, C = X.shape[0], self.hidden_dim, self.output_dim

    # 1. Forward pass
    H_lin = np.matmul(X, self.params['W1']) + self.params['b1']
    if act == 'relu':
      H_act = self.relu(H_lin)
    elif act == 'sigmoid':
      H_act = self.sigmoid(H_lin)

    assert H_act.shape == (N, H), "Hidden layer dimension mismatch"

    O = np.matmul(H_act, self.params['W2']) + self.params['b2']
    assert O.shape == (N, C), "Output layer dimension mismatch"
    
    # Remove numerical instability
    O -= np.max(O, axis=1, keepdims=True)
    smx = self.softmax(O)

    if y is None:
      return smx
    
    # 2. Compute loss
    loss = np.sum(-np.log(smx[np.arange(N), y])) / N
    loss += reg * (np.sum(self.params['W1']**2) + np.sum(self.params['W2']**2))

    # 3. Compute gradients of parameters
    grads = {}

    smx[np.arange(N), y] -= 1
    dO = smx / N

    # [H, C] = [H, N] x [N, C]
    grads['W2'] = np.matmul(H_act.T, dO)
    grads['W2'] += 2 * reg * self.params['W2']
    grads['b2'] = np.sum(dO, axis=0)

    # # [N, H] = [N, C] x [C, H]
    dH_act = np.matmul(dO, self.params['W2'].T)
    if act == 'relu':
      dH_lin = dH_act * (H_act > 0)
    elif act == 'sigmoid':
      dH_lin = dH_act * ((1-H_act) * H_act)

    # [D, H] = [D, N] x [N, H]
    grads['W1'] = np.matmul(X.T, dH_lin)
    grads['W1'] += 2 * reg * self.params['W1']
    grads['b1'] = np.sum(dH_lin, axis=0)

    for param, value in self.params.items():
      if param in grads:
        assert value.shape == grads[param].shape, \
        'Array <-> Gradient Shape Mismatch ({})'.format(param)

    return loss, grads
    
  def train(self, X, y, batch_sz=100, epochs=10, reg=0, lr=0.1, decay=False, lr_decay=0.9, patience=10):
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
    N = X.shape[0]
    loss_history = []

    with trange(epochs) as t:
      no_improve = 0
      for epoch in t:
        t.set_description('Runing epoch {}/{}'.format(epoch+1, epochs))

        # Run SGD(Stochastic Gradient Descent)
        batch_indices = np.random.choice(N, batch_sz)
        X_batch = X[batch_indices]
        y_batch = y[batch_indices]

        loss, grads = self.loss(X_batch, y_batch, reg=reg)
        
        # Learning rate decay
        if decay and len(loss_history) != 0 and loss_history[-1] <= loss:
          no_improve += 1
          if no_improve >= patience:
            no_improve = 0
            lr *= lr_decay

        loss_history.append(loss)

        # Update parameters
        for param in self.params:
          self.params[param] -= lr * grads[param]
    
    return loss_history
  
  def predict(self, X):
    """
    Predict labels of input data array
    """
    smx = self.loss(X)
    return np.argmax(smx, axis=1)
