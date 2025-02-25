import torch

class Network(torch.nn.Module):
    def __init__(self, layer_dims, activations):
        super(Network, self).__init__()
        self.layers = torch.nn.ModuleList()
        for i in range(len(layer_dims) - 1):
            self.layers.append(torch.nn.Linear(layer_dims[i], layer_dims[i + 1]))
        self.activations = activations

    def forward(self, x):
        for layer, activation in zip(self.layers, self.activations):
            x = layer(x)
            if activation == 'sigmoid':
                x = torch.sigmoid(x)
            elif activation == 'relu':
                x = torch.relu(x)
        return x
