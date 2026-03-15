import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import joblib
import matplotlib.pyplot as plt
df = pd.read_csv("cognitive_load_dataset_3000.csv")
df.head()
X = df.drop("cognitive_load", axis=1)
y = df["cognitive_load"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)
print(classification_report(y_test, y_pred))
df['cognitive_load'].value_counts().plot(kind='bar')
plt.title("Cognitive Load Distribution")
plt.xlabel("Load Level")
plt.ylabel("Count")
plt.show()

joblib.dump(model, "cognitive_load_model.pkl")
print("Model Saved Successfully")
