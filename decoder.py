import torch
import torch.nn.functional as F
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 activation,
                 last_activation,
                 bias,
                 num_layers=6,
                 hidden_dim=128,
                 ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = self.fetch_activation(activation)
        self.last_activation = self.fetch_activation(last_activation)
        self.bias = bias
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        layers = []
        for i in range(self.num_layers):
            if i == 0:
                layers.append(nn.Linear(self.input_dim, self.hidden_dim, bias=self.bias))
            else:
                layers.append(nn.Linear(self.hidden_dim, self.hidden_dim, bias=self.bias))
        self.layers = nn.ModuleList(layers)
        self.out_layer = nn.Linear(self.hidden_dim, self.output_dim, bias=self.bias)

    def forward(self, x):
        for i, l in enumerate(self.layers):
            if i == 0:
                h = self.activation(l(x))
            else:
                h = self.activation(l(h))

        out = self.last_activation(self.out_layer(h))
        # out = torch.sigmoid(out)
        return out

    def fetch_activation(self, activation):
        if activation == "Relu":
            return nn.ReLU()
        elif activation == "Linear":
            return nn.Identity()
        elif activation == "Sigmoid":
            return nn.Sigmoid()
        elif activation == "Tanh":
            return nn.Tanh()
        else:
            print(f"Unknown activation {activation}")
            return nn.ReLU()