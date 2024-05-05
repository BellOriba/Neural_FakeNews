import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

def splitDataSet(path):
    data = pd.read_csv(path)
    data['label'] = data['label'].map({'Fake': 0, 'Real': 1})
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    return train_data, test_data

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

if __name__ == "__main__":
    print("Running preprocess.py")
