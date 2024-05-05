import torch
import torch.nn as nn
import torch.optim as optim

class FakeNewsClassifier(nn.Module):
    def __init__(self, input_size):
        super(FakeNewsClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def lossOptimizer(model):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    return criterion, optimizer

if __name__ == "__main__":
    pass
