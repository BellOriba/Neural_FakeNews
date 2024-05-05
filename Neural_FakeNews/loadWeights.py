import torch
import torch.nn as nn

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

input_size = 52816
model = FakeNewsClassifier(input_size)

model_state_dict = {}
with open('model_weights.txt', 'r') as file:
    lines = file.readlines()
    layer_name = None
    weights = None
    for line in lines:
        if line.startswith('Layer'):
            if layer_name is not None and weights is not None:
                model_state_dict[layer_name] = torch.tensor(weights)
            layer_name = line.split(':')[1].strip()
            weights = []
        elif line.startswith('Weights'):
            pass
        elif line.strip() == '':
            pass
        else:
            weights.append([float(value) for value in line.strip().split()])

# Modelo Carregado:
model.load_state_dict(model_state_dict)
