from engine import Value
import pandas as pd
import math
import numpy as np
import random
import pickle

e = math.e

df = pd.read_csv('Dataset.csv')


X = df.iloc[:, :-1].values.astype(float)
Y = df.iloc[:, -1].values.astype(int)

X_train = X[:8000]
X_test  = X[8000:]
y_train = Y[:8000]
y_test  = Y[8000:]

# z-score normalization
means = X_train.mean(axis=0)
stds = X_train.std(axis=0)
stds[stds == 0] = 1.0
X_train = ((np.array(X_train) - means) / stds).tolist() 
X_test = ((np.array(X_test) - means) / stds).tolist() 

parameters = [Value(random.uniform(-0.01, 0.01)) for _ in range(len(X_train[0]) + 1)]

def sigmoid(z: Value):
    return Value(1/(1 + e**(-z.data)), _children=(z,), _op='sigmoid')

def predict(parameters, x):
    z = Value(0)
    for j in range(len(x)):
        z = z + parameters[j]*x[j]
    z = z + parameters[-1]
    return sigmoid(z)

def loss(parameters, X, Y):
    total_loss = Value(0)
    eps = 1e-15
    for x,y in zip(X, Y):
        f = predict(parameters, x)
        if y == 1:
            child = f + eps
            loss_i = -Value(math.log(child.data), _children=(child,), _op='ln')
        else:
            child = (Value(1) - f) + eps
            loss_i = -Value(math.log(child.data), _children=(child,), _op='ln')
        total_loss = total_loss + loss_i
    avg_loss = total_loss / len(X)
    return avg_loss

def train(epochs=1000, alpha=0.01):
    for epoch in range(epochs):
        total_loss = loss(parameters, X_train, y_train)
        total_loss.backward()
        for idx in range(len(parameters)):
            parameters[idx].data = parameters[idx].data - alpha * parameters[idx].grad
            parameters[idx].grad = 0.0
        if epoch % 10 == 0:
            print(f"epoch {epoch} loss {total_loss.data}")

train()

bundle = {
    'model': parameters,
    'X_test': X_test,
    'y_test': y_test,
    'means' : means,
    'stds' : stds
}

with open('manually_trained_model.bin', 'wb') as file:
    pickle.dump(bundle, file)

print("Model trained and saved successfully!")
