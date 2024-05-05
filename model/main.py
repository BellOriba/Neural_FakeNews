import time # Só pra gravar o tempo de computação
import preprocess
import model as mdl
import torch
from torch.utils.data import DataLoader
from sklearn.feature_extraction.text import CountVectorizer

train_data, test_data = preprocess.splitDataSet("./data/fake_and_real_news.csv")

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

model = mdl.FakeNewsClassifier(input_size)
criterion, optimizer = mdl.lossOptimizer(model)

# Define batch size
batch_size = 64

# Create datasets and data loaders
train_dataset = preprocess.NewsDataset(train_data, vectorizer)
test_dataset = preprocess.NewsDataset(test_data, vectorizer)

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

print("inputSize: ", input_size)
