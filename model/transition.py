from torch import nn as nn


class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_prob=None):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_sizes[0], device='cuda'))
        # self.dropout = nn.Dropout(dropout_prob)

        for i in range(1, len(hidden_sizes)):
            self.layers.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i], device='cuda'))

        self.output_layer = nn.Linear(hidden_sizes[-1], output_size, device='cuda')
        self.leaky_relu = nn.LeakyReLU(0.01)

    def forward(self, x):
        for layer in self.layers:
            x = self.leaky_relu(layer(x))
            # x = self.dropout(x)  # Apply dropout after the activation function
        x = self.output_layer(x)
        return x
