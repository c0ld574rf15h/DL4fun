import numpy as np

class NeuralNet():
  def __init__(self, input_dim, hidden_dim, output_dim):
    self.params = {}
    self.params['W1'] = np.random.randn(input_dim, hidden_dim)
    self.params['b1'] = np.zeros(hidden_dim)
    self.params['W2'] = np.random.randn(hidden_dim, output_dim)
    self.params['b2'] = np.zeros(output_dim)
  
  def forward(self, X):
    print('Input size: {}'.format(X.shape))

  def train(self):
    pass
  
  def predict(self):
    pass