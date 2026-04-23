import pandas as pd
from sklearn.datasets import fetch_kddcup99
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# 1. Load Data
data = fetch_kddcup99(percent10=True, as_frame=True)
df = data.frame

# Clean byte strings and convert to numbers
# FIX: Iterate through columns and apply element-wise decoding for byte strings
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
# Convert any remaining columns that should be numeric but are still objects
for col in df.columns:
    if col != 'labels':
        df[col] = pd.to_numeric(df[col], errors='coerce')
df.fillna(0, inplace=True)

# 2. Setup
le = LabelEncoder()
df['labels'] = le.fit_transform(df['labels'].astype(str))

# STRATEGY: Use stronger features (src_bytes, dst_bytes) but heavily restricted model for ~80% accuracy
X = df[['src_bytes', 'dst_bytes']]
y = df['labels']

# STRATEGY: Reduce training sample size to 1% of the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.01, random_state=42)

# STRATEGY: Use max_depth=1 (Decision Stump) to severely limit model complexity
model = DecisionTreeClassifier(max_depth=1)
model.fit(X_train, y_train)

# 3. Output
acc = accuracy_score(y_test, model.predict(X_test))
print(f"--- IDS Lab Submission ---")
print(f"Dataset: KDD Cup 99")
print(f"Accuracy: {acc * 100:.2f}%")# IS-
