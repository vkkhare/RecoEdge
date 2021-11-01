import torch
from fedrec.utilities import registry

registry.load("sigmoid_layer", "relu")(torch.nn.ReLU)
registry.load("sigmoid_layer", "sigmoid")(torch.nn.Sigmoid)
