import pickle
from sklearn.metrics import accuracy_score, classification_report

with open('model_complete.bin', 'rb') as file:
    data = pickle.load(file)

model = data['model']
X_test = data['X_test']
y_test = data['y_test']

y_pred = model.predict(X_test)

print(model.coef_)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))