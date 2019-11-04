import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class ApproximateEncoder(nn.Module):
    def __init__(self):
        super(ApproximateEncoder, self).__init__()
        
        self.b1 = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )
        
        self.b2 = nn.Sequential(
            nn.Conv2d(32, 64, 3),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )
        
        self.b3 = nn.Sequential(
            nn.Conv2d(64, 128, 3),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )
        
        
        self.b4 = nn.Sequential(
            nn.Conv2d(128, 128, 2),
            nn.ReLU(),
        )
        
        self.fc = nn.Linear(128, 128)
        
    def forward(self, x):
        b_size = x.shape[0]
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.fc(x.view(b_size, -1))
        return x
    
    def train(self, G, samples=1000, epochs=100, batch_size=50, lr=0.01, device='cuda:0'):
        G = G.to(device)
        
        target = torch.rand(samples, 128).to(device)
        target_y = torch.LongTensor(samples).random_(0, 10).to(device)
        
        X = G(target, target_y)
        E_optimizer = optim.Adam(self.parameters(), lr=lr)
        E_crit = nn.MSELoss()
        
        dataloader = DataLoader(TensorDataset(X, target), batch_size=batch_size, shuffle=True)
    
        for _ in range(epochs):
            for inpt, label in dataloader:
                self.zero_grad()
                output = self(inpt)
                loss = E_crit(output, label)
                loss.backward()
                E_optimizer.step()