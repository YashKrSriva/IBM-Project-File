import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import joblib

# Load dataset
df = pd.read_csv("data/sat_collision.csv")

# Select features & target
X = df[['Relative_Distance','Relative_Speed','Debris_Size','Approach_Angle','Debris_Type']]
y = df['Risk']

# Preprocessing for categorical feature
preprocess = ColumnTransformer(
    [('cat', OneHotEncoder(handle_unknown='ignore'), ['Debris_Type'])],
    remainder='passthrough'
)

X_trans = preprocess.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_trans, y, test_size=0.2, random_state=42, stratify=y)

# I trained two models to compare their performance
lr = LogisticRegression(max_iter=200)
rf = RandomForestClassifier(n_estimators=120, random_state=42)

lr.fit(X_train, y_train)
rf.fit(X_train, y_train)

# Making predictions
y_pred_lr = lr.predict(X_test)
y_pred_rf = rf.predict(X_test)

# Print results
print("=== Logistic Regression ===")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print("F1 Score:", f1_score(y_test, y_pred_lr))

print("\n=== Random Forest ===")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("F1 Score:", f1_score(y_test, y_pred_rf))
print("\nClassification Report:\n", classification_report(y_test, y_pred_rf))

# Save best model
joblib.dump((rf, preprocess), "models/sat_rf_model.pkl")
print("\nModel saved to: models/sat_rf_model.pkl")
