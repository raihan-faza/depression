import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.optim as optim
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# importing dataset
df = pd.read_csv("data.csv")
print(df.info())

# checking null values
print(df.isna().sum())

# looks like there is a nan value in financial stress, well its only 3 maybe we should delete that
df.dropna(inplace=True)
print(df.isna().sum())

# changin object type data to numerical data
encoder = LabelEncoder()
object_columns = df.select_dtypes(include=["object"]).columns
print(df[object_columns].info())
for column in object_columns:
    df[column + "_encoded"] = encoder.fit_transform(df[column])
    df.drop(columns=column, inplace=True)
print(df.info())
print(df["Depression"])
# eda tipis - tipis
# who depressed more? men or women?
# men udah percaya aja, jk
"""
sns.countplot(data=df, x="Gender")
plt.title("Count Plot of Categories")
plt.xlabel("Gender")
plt.ylabel("Total")
plt.show()
"""

# how does each data correlates
"""
sns.heatmap(df.corr(), annot=True, cmap="crest")
plt.show()
"""


# making the model


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.lahh = nn.Sequential(
            nn.Linear(in_features=17, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=1),  # For binary classification
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.lahh(x)
        return logits


class MyDS(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x.to_numpy(), dtype=torch.float32)
        self.y = torch.tensor(
            y.to_numpy(), dtype=torch.float32
        )  # Changed to float32 for BCEWithLogitsLoss

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# split dataset
x = df.iloc[:, :-1]
y = df.iloc[:, -1]
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)

# forming new dataset to load to pytorch
train_ds = MyDS(xtrain, ytrain)
test_ds = MyDS(xtest, ytest)
train_loader = DataLoader(train_ds, batch_size=10, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=10, shuffle=True)

# train the model
model = NeuralNetwork()
num_epochs = 100
criterion = nn.BCEWithLogitsLoss()  # For binary classification
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    for batch_idx, (data, target) in enumerate(
        tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
    ):
        optimizer.zero_grad()  # Reset gradients
        output = model(data)  # Forward pass
        loss = criterion(
            output.squeeze(), target
        )  # Squeeze the output to match target shape
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights

# Save the model
torch.save(model, "model.pth")

# evaluate model
model.eval()
y_true, y_pred = [], []

with torch.no_grad():  # No need to track gradients
    for inputs, labels in test_loader:
        outputs = model(inputs)
        # Apply sigmoid and threshold to get predictions
        predictions = torch.sigmoid(outputs).squeeze() > 0.5  # Binary thresholding
        y_true.extend(labels.numpy())
        y_pred.extend(predictions.numpy())

accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy: {accuracy}")
