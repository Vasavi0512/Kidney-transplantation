# Kidney-transplantation
Welcome to the Bio-and-Ethics wiki!
BIO - Increased the datapoints of the dataset. We didnt get the exact dataset for our project so we are proceeding with the available dataset. We have started training the ML model using LightGBM. For now the model is underfit. We are increasing the datapoints some more and we are trying to get accurate accuracy value.
ETHICS - We have collected research papers and clinical trial papers to collect ethical values and IPR values 

1st review ppt - 
[Organtransplantation-AIM-A-3.pptx](https://github.com/user-attachments/files/19145360/Organtransplantation-AIM-A-3.pptx)

Code using Light GBM model

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 🔹 Load the dataset
file_path = "/Users/sahassrak/Documents/organ_matchmaking.xlsx" # Update path
df = pd.read_excel(file_path)

# 🔹 Clean column names (remove special characters)
df.columns = df.columns.str.replace(r'[^\w]', '', regex=True)  # Removes special characters

# 🔹 Print columns for verification
print("Columns in dataset:", df.columns.tolist())

# 🔹 Define target variable
target_column = "Received_Transplant"

# 🔹 Check if target column exists
if target_column not in df.columns:
    raise ValueError(f"⚠ Target column '{target_column}' not found in dataset!")

# 🔹 Handle categorical variables
categorical_columns = [
    "Patient_Gender", "Race", "Underlying_Disease", "Has_Diabetes",
    "Has_Prior_Transplant", "Blood_Group"
]

df[categorical_columns] = df[categorical_columns].astype("category")

# 🔹 Convert 'Priority_Score' to numeric (fix dtype issue)
df["Priority_Score"] = pd.to_numeric(df["Priority_Score"], errors="coerce")

# 🔹 Fill missing values (numerical)
df.fillna(df.median(numeric_only=True), inplace=True)

# 🔹 Define features & target
X = df.drop(columns=[target_column])
y = df[target_column]

# 🔹 Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔹 Train LightGBM Model
model = lgb.LGBMClassifier()
model.fit(X_train, y_train)

# 🔹 Predictions
y_pred = model.predict(X_test)

# 🔹 Evaluate Model
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Matchmaking Model Accuracy: {accuracy:.2%}")
