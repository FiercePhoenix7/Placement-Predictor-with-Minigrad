import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

df = pd.read_csv('Dataset.csv')

X = df.iloc[:, :-1].values
Y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

model = LogisticRegression(random_state=9, max_iter=10000).fit(X_train, y_train)

bundle = {
    'model': model,
    'X_test': X_test,
    'y_test': y_test,

}

with open('model_complete.bin', 'wb') as file:
    pickle.dump(bundle, file)

print("Model trained and saved successfully!")

