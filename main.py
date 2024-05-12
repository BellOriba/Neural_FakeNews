import pandas as pd
import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import tensorflow as tf
from tensorflow.keras import layers, Model

data = pd.read_csv("./data/fake_and_real_news.csv")

data['label'] = data['label'].map({'Fake': 0, 'Real': 1})
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

class NewsDataset(tf.keras.utils.Sequence):
    def __init__(self, dataframe, vectorizer, batch_size=64):
        self.dataframe = dataframe
        self.vectorizer = vectorizer
        self.batch_size = batch_size

    def __len__(self):
        return len(self.dataframe) // self.batch_size

    def __getitem__(self, idx):
        batch_data = self.dataframe.iloc[idx * self.batch_size:(idx + 1) * self.batch_size]
        texts = batch_data['Text']
        labels = batch_data['label']
        vectorized_texts = self.vectorizer.transform(texts).toarray()
        return vectorized_texts, labels.values

class FakeNewsClassifier(Model):
    def __init__(self, input_size):
        super(FakeNewsClassifier, self).__init__()
        self.fc1 = layers.Dense(64, activation='relu')
        self.fc2 = layers.Dense(32, activation='relu')
        self.fc3 = layers.Dense(2)

    def call(self, inputs):
        x = self.fc1(inputs)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

start_time = time.time()

vectorizer = CountVectorizer()
train_text = train_data['Text']
vectorizer.fit(train_text)

input_size = len(vectorizer.get_feature_names_out())

test_text = test_data['Text']
test_features = vectorizer.transform(test_text)
test_labels = test_data['label']

model = FakeNewsClassifier(input_size)

criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam()

batch_size = 64

train_dataset = NewsDataset(train_data, vectorizer, batch_size=batch_size)
test_dataset = NewsDataset(test_data, vectorizer, batch_size=batch_size)

num_epochs = 10
for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, labels in train_dataset:
        with tf.GradientTape() as tape:
            outputs = model(inputs.astype(np.float32))
            loss = criterion(labels, outputs)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        running_loss += loss.numpy()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_dataset)}')

correct = 0
total = 0
for inputs, labels in test_dataset:
    outputs = model(inputs.astype(np.float32))
    predicted = np.argmax(outputs, axis=1)
    total += labels.shape[0]
    correct += np.sum(predicted == labels)

print(f'Accuracy on test set: {(100 * correct / total):.2f}%')

end_time = time.time()
total_time = end_time - start_time
print(f"Total training time: {total_time} seconds")

model_weights = {var.name: var.numpy() for var in model.trainable_variables}
with open('model_weights.txt', 'w') as file:
    for name, param in model_weights.items():
        file.write(f"Layer: {name}\n")
        file.write("Weights:\n")
        file.write(str(param))  
        file.write("\n\n")

