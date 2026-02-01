import torch.nn as nn
import torch

class Generator(nn.Module):
    def __init__(self, n_layers, n_hidden, latent_dim, input_dim):
        super(Generator, self).__init__()

        self.input_layer = nn.ModuleList()
        self.hidden_layers = nn.ModuleList()
        self.output_layer = nn.ModuleList()

        self.activations = nn.ModuleList()

        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        
        if n_layers > 0:
            self.input_layer.append(nn.Linear(latent_dim, n_hidden))
            self.activations.append(nn.LeakyReLU(0.2, inplace=True))
            for i in range(n_layers):
                self.hidden_layers.append(nn.Linear(n_hidden, n_hidden))
                self.activations.append(nn.LeakyReLU(0.2, inplace=True))
            self.output_layer.append(nn.Linear(n_hidden, input_dim))

        if n_layers == 0:
            self.input_layer.append(nn.Linear(latent_dim, input_dim))

    def forward(self, z):        
        z = torch.squeeze(z, (2,3))
        if self.n_layers > 0:
            z = self.input_layer[0](z) 
            z = self.activations[0](z)
            for i in range(self.n_layers):
                z = self.hidden_layers[i](z)
                z = self.activations[i+1](z)
            z = self.output_layer[0](z)

        if self.n_layers == 0:
            z = self.input_layer[0](z) 
            
        return z

class Discriminator(nn.Module):
    def __init__(self, n_layers, n_hidden, latent_dim, input_dim):
        super(Discriminator, self).__init__()

        self.input_layer = nn.ModuleList()
        self.hidden_layers = nn.ModuleList()
        self.output_layer = nn.ModuleList()

        self.activations = nn.ModuleList()
        
        if n_layers > 0:
            self.input_layer.append(nn.Linear(latent_dim, n_hidden))
            self.activations.append(nn.LeakyReLU(0.2, inplace=True))
            for i in range(n_layers):
                self.hidden_layers.append(nn.Linear(n_hidden, n_hidden))
                self.activations.append(nn.LeakyReLU(0.2, inplace=True))
            self.output_layer.append(nn.Linear(n_hidden, input_dim))

        if n_layers == 0:
            self.input_layer.append(nn.Linear(latent_dim, input_dim))

    def forward(self, z):
            
        return z

    
