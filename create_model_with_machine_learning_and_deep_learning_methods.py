# -*- coding: utf-8 -*-
"""
Created on Sat Dec 30 14:36:16 2023

@author: BatuhanYILMAZ
"""

from pandas import read_csv
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import precision_score,recall_score,f1_score
import pandas as pd
from sklearn.metrics import classification_report

df1 = read_csv("project_features.csv")


sns.countplot(x="label", data=df1,palette="Set3")
plt.title("Labels Count")
plt.show()

models=dict()
models['KNN'] = KNeighborsClassifier(n_neighbors=7)
models['DT'] = DecisionTreeClassifier()
models['SVC'] = SVC()
models['GNB'] = GaussianNB()
# ensemble models
models['BAG'] = BaggingClassifier(n_estimators=100)
models['RF'] = RandomForestClassifier(n_estimators=100)
models['ExT'] = ExtraTreesClassifier(n_estimators=100)
models['GBC'] = GradientBoostingClassifier(n_estimators=100)
x = df1.iloc[:,:6]
y = df1.iloc[:,-1]
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=0)

def ml(df,models):
    x = df.iloc[:,:6]
    y = df.iloc[:,-1]
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=0)
    def accuracy(trainX, trainy, testX, testy, models):
        acc = list()
        for name, model in models.items():
            model.fit(trainX, trainy)
            yhat = model.predict(testX)
            acc.append(accuracy_score(testy, yhat) * 100.0)
        return acc
    def recall(trainX, trainy, testX, testy, models):
    	# fit the model
        rcc = list()
        for name, model in models.items():
            model.fit(trainX, trainy)
            yhat = model.predict(testX)
            rcc.append(recall_score(testy, yhat ,average='micro') * 100.0)
        return rcc
    def f1(trainX, trainy, testX, testy, models):
        f = list()
        for name, model in models.items():
            model.fit(trainX, trainy)
            yhat = model.predict(testX)
            f.append(f1_score(testy, yhat, average='micro') * 100.0)
        return f
    
    
    def prec(trainX, trainy, testX, testy, model):
        prec = list()
        for name, model in models.items():
            model.fit(trainX, trainy)
            yhat = model.predict(testX)
            prec.append(precision_score(testy, yhat, average='micro') * 100.0)
        return prec
    
    def print_metrics(trainX, trainy, testX, testy, models):
        for name, model in models.items():
            model.fit(trainX, trainy)
            yhat = model.predict(testX)

            accuracy = accuracy_score(testy, yhat) * 100.0
            precision = precision_score(testy, yhat, average='micro') * 100.0
            recall = recall_score(testy, yhat, average='micro') * 100.0
            f1 = f1_score(testy, yhat, average='micro') * 100.0

            print(f"Metrics for {name} Model:")
            print(f"Accuracy: {accuracy:.2f}%")
            print(f"Precision: {precision:.2f}%")
            print(f"Recall: {recall:.2f}%")
            print(f"F1 Score: {f1:.2f}%\n{'='*40}")

    print_metrics(x_train, y_train, x_test, y_test, models)
    
    Accresults = accuracy(x_train, y_train, x_test, y_test , models)
    Precresults = prec(x_train, y_train, x_test, y_test , models)
    f1results = f1(x_train, y_train, x_test, y_test , models)
    Recallresults = recall(x_train, y_train, x_test, y_test , models)
    return Accresults,Precresults,f1results,Recallresults


from joblib import dump
random_forest_model = models['RF']

# Save the Random Forest model to a file
#dump(random_forest_model, 'random_forest_model.joblib')
 
# Call the function to print metrics
ml(df1, models)

def plot_together(df):
    #creating a dataframe and comparing the scores an plotting 
    compareResult = pd.DataFrame(list(models.keys()))
    compareResult["Acc"] = (ml(df,models))[0]
    compareResult["prec"] = (ml(df,models))[1]
    compareResult["F1"] = (ml(df,models))[2]
    compareResult["Recall"] = (ml(df,models))[3]
    compareResult.plot(kind="bar");
    plt.xlabel(list(models.keys()))

#Making prediction with the best model ExtraTreesClassifier by looking at the graph
best_model = RandomForestClassifier()
best_model.fit(x_train, y_train)
pred = best_model.predict(x_test)

from sklearn.metrics import confusion_matrix 
conf_matrix = confusion_matrix(y_true=y_test, y_pred = pred)
#
# Print the confusion matrix using Matplotlib
#
fig, ax = plt.subplots(figsize=(5, 5))
ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')

plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()

plot_together(df1)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Convert data to PyTorch tensors
X_tensor = torch.tensor(df1.iloc[:, :6].values, dtype=torch.float32)
y_tensor = torch.tensor(df1['label'].values, dtype=torch.long)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

# Define a  neural network
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, dropout_rate=0.5):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(hidden_size2, output_size)

        # Initialize weights using Kaiming initialization
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.kaiming_normal_(self.fc3.weight)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

# Hyperparameters
input_size = 6
hidden_size1 = 128
hidden_size2 = 64
output_size = 2  # Assuming binary classification
dropout_rate = 0.5

# Initialize the neural network
model = NeuralNetwork(input_size, hidden_size1, hidden_size2, output_size, dropout_rate)

# Loss function and optimizer with learning rate scheduling
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

# Train the neural network
num_epochs = 2500
for epoch in range(num_epochs):
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    if (epoch + 1) % 5 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Make predictions using the trained model
with torch.no_grad():
    predicted = model(X_test).argmax(dim=1)

# Evaluate the performance
accuracy = accuracy_score(y_test.numpy(), predicted.numpy())
precision = precision_score(y_test.numpy(), predicted.numpy(), average='micro')
recall = recall_score(y_test.numpy(), predicted.numpy(), average='micro')
f1 = f1_score(y_test.numpy(), predicted.numpy(), average='micro')

print(f'Accuracy: {accuracy * 100:.2f}%')
print(f'Precision: {precision * 100:.2f}%')
print(f'Recall: {recall * 100:.2f}%')
print(f'F1 Score: {f1 * 100:.2f}%')

