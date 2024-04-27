import pandas as pd
import time # Só pra gravar o tempo de computação

# Bibliotecas para o exemplo do ChatGPT
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

data = pd.read_csv("./data/fake_and_real_news.csv")

# Converte a label de string para int, fica mais fácil de trabalhar
data['label'] = data['label'].map({'Fake': 0, 'Real': 1})
# Divide o dataset entre Dados de Treino e Dados de Teste (80%/20%)
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# ------------------------------------------------
# Código Exemplo do ChatGPT:
# Define a custom Dataset class:
class NewsDataset(Dataset):
    def __init__(self, dataframe, vectorizer):
        self.dataframe = dataframe
        self.vectorizer = vectorizer

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        text = self.dataframe.iloc[idx]['Text']
        label = self.dataframe.iloc[idx]['label']
        vectorized_text = self.vectorizer.transform([text]).toarray().squeeze()
        return vectorized_text, label

# Define the neural network model:
class FakeNewsClassifier(nn.Module):
    def __init__(self, input_size):
        super(FakeNewsClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)  # 2 classes: Fake or Real

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Record the starting time
start_time = time.time()

# Instantiate the model
vectorizer = CountVectorizer()
train_text = train_data['Text']
vectorizer.fit(train_text)

input_size = len(vectorizer.get_feature_names_out())
test_text = test_data['Text']
test_features = vectorizer.transform(test_text)
test_labels = test_data['label']

model = FakeNewsClassifier(input_size)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Define batch size
batch_size = 64

# Create datasets and data loaders
train_dataset = NewsDataset(train_data, vectorizer)
test_dataset = NewsDataset(test_data, vectorizer)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Train the model
num_epochs = 10
for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, labels in train_loader:
        # Forward pass
        outputs = model(inputs.float())
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader)}')

# Evaluate the model on the test set:
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs.float())
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy on test set: {(100 * correct / total):.2f}%')

# Calcular o tempo total
end_time = time.time()
total_time = end_time - start_time
print(f"Total training time: {total_time} seconds")

# Salvar os pesos da rede neural em um arquivo txt
model_state_dict = model.state_dict()

with open('model_weights.txt', 'w') as file:
    for name, param in model_state_dict.items():
        file.write(f"Layer: {name}\n")
        file.write("Weights:\n")
        file.write(str(param.cpu().numpy()))  # Convert tensor to numpy array for saving
        file.write("\n\n")

