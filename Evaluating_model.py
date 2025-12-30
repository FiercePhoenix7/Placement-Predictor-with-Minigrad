import pickle
import math
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

e = math.e

with open('ScikitLearn_model.bin', 'rb') as file:
    data = pickle.load(file)

model = data['model']
X_test = data['X_test']
y_test = data['y_test']

y_pred = model.predict(X_test)

print(model.coef_)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy*100:.4f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print('\n\n\t-----------------------------------------\n\n')

with open('manually_trained_model.bin', 'rb') as file:
    data = pickle.load(file)

parameters = data['model']
X_test = data['X_test']
y_test = data['y_test']
means = np.array(data['means'])
stds = np.array(data['stds'])

parameters = [parameters[i].data for i in range(len(parameters))]

def predict(parameters, x, normalized=True):
    if not normalized:
        x = ((np.array(x) - means) / stds).tolist()

    z = 0
    for j in range(len(x)):
        z += parameters[j]*x[j]
    z = z + parameters[-1]

    f = 1/(1 + e**(-z))
    threshold = 0.5
    if f >= threshold:
        return 1
    else:
        return 0
    
y_pred = [predict(parameters, x) for x in X_test]

print(parameters)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy*100:.4f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# print(predict(parameters, [7.5,1,1,1,65,4.4,0,0,61,79,], False))

