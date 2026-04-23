import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# =========================
# 1. LOAD DATASET
# =========================

# Fixed: Pointing to an existing file in the Colab environment
file_path = "/content/sample_data/california_housing_train.csv"

df = pd.read_csv(file_path)

print("Dataset Loaded Successfully!")
print("Shape:", df.shape)

# =========================
# 2. CLEAN DATA
# =========================

df.columns = df.columns.str.strip()
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

print("Data cleaned!")

# =========================
# 3. HANDLE LABEL COLUMN
# =========================

# For California Housing, let's create a binary label based on median_house_value
# (e.g., 1 if above median, 0 if below) to make the classifier work.
label_col = 'high_value_target'
median_val = df['median_house_value'].median()
df[label_col] = (df['median_house_value'] > median_val).astype(int)

# Drop the original continuous target
df = df.drop('median_house_value', axis=1)

# =========================
# 4. ENCODE CATEGORICAL DATA
# =========================

df = pd.get_dummies(df)
print("Categorical features encoded!")

# =========================
# 5. SPLIT FEATURES & LABELS
# =========================

X = df.drop(label_col, axis=1)
y = df[label_col]

# =========================
# 6. TRAIN-TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 7. FEATURE SCALING
# =========================

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Data scaled!")

# =========================
# 8. TRAIN MODEL
# =========================

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("Model trained!")
print("Accuracy:", accuracy_score(y_test, model.predict(X_test)))
