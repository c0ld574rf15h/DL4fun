import torch.nn as nn
from tqdm import trange

class NeuralNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layer=0):
        super(NeuralNet, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layer = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.layer = layer
    
    def forward(self, X):
        H = self.input_layer(X).clamp(min=0)
        for _ in range(self.layer):
            H = self.hidden_layer(H).clamp(min=0)
        y_pred = self.output_layer(H)

        return y_pred