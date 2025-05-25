import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Load the data (replace with your file path)
df = pd.read_csv("Disease.csv")
## Explore Data (EDA)
print("Data shape:", df.shape)
print("Sample rows:")
print(df.head())

# Count diseases
print("\nDisease value counts:")
print(df['Disease'].value_counts())

# Count NaNs per column
print("\nMissing values per column:")
print(df.isnull().sum())

# Visualize top 10 common diseases
plt.figure(figsize=(10,5))
df['Disease'].value_counts().head(10).plot(kind='barh')
plt.title("Top 10 Most Frequent Diseases")
plt.xlabel("Count")
plt.show()
# Clean and combine all symptoms
symptom_columns = [f"Symptom_{i}" for i in range(1, 18)]
df["all_symptoms"] = df[symptom_columns].apply(
    lambda x: [s.strip() for s in x if pd.notna(s)], axis=1
)

# Create a unique list of all symptoms
unique_symptoms = sorted({symptom for row in df["all_symptoms"] for symptom in row})
# Create binary vectors for each row
def encode_symptoms(symptom_list, all_symptoms):
    return [1 if s in symptom_list else 0 for s in all_symptoms]

df["symptom_vector"] = df["all_symptoms"].apply(lambda x: encode_symptoms(x, unique_symptoms))

# Expand into separate columns
df_encoded = pd.DataFrame(df["symptom_vector"].tolist(), columns=unique_symptoms)
df_encoded["prognosis"] = df["Disease"]

X = df_encoded.drop("prognosis", axis=1)
y = df_encoded["prognosis"]
print("Feature matrix shape (X):", X.shape)
print("Target vector shape (y):", y.shape)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = MultinomialNB()
model.fit(X_train, y_train)
# List of all possible symptoms
all_symptom_list = list(X.columns)

# User symptoms
user_symptoms = ['skin_rash','chills']

# Convert user symptoms to one-hot vector
def symptoms_to_vector(user_symptoms, all_symptoms):
    return [1 if s in user_symptoms else 0 for s in all_symptoms]

input_vector = [symptoms_to_vector(user_symptoms, all_symptom_list)]
prediction = model.predict(input_vector)[0]

print("Predicted Disease:", prediction)

y_pred = model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nAccuracy Score:", accuracy_score(y_test, y_pred))

import pickle

# Save the trained model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save the symptom list
with open("symptoms_list.pkl", "wb") as f:
    pickle.dump(all_symptom_list, f)

