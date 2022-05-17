import torch
import torch.nn as nn
import torch.nn.functional as F

device = 'cuda:0'

# Model architecture
class deepAE(nn.Module):
    def __init__(self):
        super(deepAE, self).__init__()
        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(in_features = 4, out_features = 4),
            nn.ELU(),
            nn.Linear(in_features = 4, out_features = 3),
            nn.ELU(),
            nn.Linear(in_features = 3, out_features = 2),
            nn.ELU()
        )

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(in_features = 2, out_features = 3),
            nn.ELU(),
            nn.Linear(in_features = 3, out_features = 4),
            nn.ELU(),
            nn.Linear(in_features = 4, out_features = 4),
            nn.ELU()
        )
    
    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class finenet(nn.Module):
    def __init__(self, modelA):
        super(finenet, self).__init__()
        self.modelA = modelA
        self.linear = nn.Linear(in_features=2, out_features=2)
        
    def forward(self,x):
        x = self.modelA(x)
        x = self.linear(x)
        return x 