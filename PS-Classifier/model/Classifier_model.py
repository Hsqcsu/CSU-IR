import torch.nn as nn
import torch

class classifymodel(nn.Module):
    def __init__(self, dim=1024,num_classes=2):
        super(classifymodel, self).__init__()

        self.norm = nn.LayerNorm(dim)
        self.linear = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.norm(x)
        x = self.linear(x)
        return x

    def load_weights(self, path):
        if path is not None:
            model_dict = torch.load(path, map_location=torch.device('cpu'))
            self.load_state_dict(model_dict)

