# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix
import joblib
import os

# Set plot style
plt.style.use('ggplot')

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

print("Loading dataset...")
# Load the dataset - using the correct path with quotes
df = pd.read_csv("C:/mlops/churn-prediction-mlops/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Display basic information
print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Info:")
df.info()

print("\nMissing values per column:")
print(df.isnull().sum())

print("\nChurn Distribution:")
print(df['Churn'].value_counts())
print(df['Churn'].value_counts(normalize=True) * 100)

# Visualize churn distribution
plt.figure(figsize=(8, 5))
sns.countplot(x='Churn', data=df)
plt.title('Customer Churn Distribution')
plt.savefig('models/churn_distribution.png')
print("Saved churn distribution plot to 'models/churn_distribution.png'")

print("\nData preprocessing...")
# Convert TotalCharges to numeric
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Check for missing values after conversion
print("Missing values after TotalCharges conversion:")
print(df.isnull().sum())

# Fill missing values - fixed warning by avoiding inplace
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].mean())

# Remove 'customerID' column
df.drop('customerID', axis=1, inplace=True)

# Convert 'Yes'/'No' values to 1/0 for the target variable
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Identify categorical and numerical columns
categorical_cols = df.select_dtypes(include=['object']).columns
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.drop('Churn')

print("Categorical columns:", list(categorical_cols))
print("Numerical columns:", list(numerical_cols))

# Split data into features and target
X = df.drop('Churn', axis=1)
y = df['Churn']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

# Fit the preprocessor to the training data
preprocessor.fit(X_train)

# Save the preprocessor for later use
joblib.dump(preprocessor, 'models/preprocessor.pkl')
print("Preprocessor saved to 'models/preprocessor.pkl'")

print("\nTraining models...")
# Define the models to train
models = {
    "Logistic_Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random_Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

results = {}
best_auc = 0
best_model_name = None
best_model = None

# Train and evaluate each model
for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Create and fit the pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    # Store results
    results[name] = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "auc": auc
    }
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Save confusion matrix plot
    cm_path = f'models/{name}_confusion_matrix.png'
    plt.savefig(cm_path)
    
    # Print results
    print(f"{name} Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  AUC: {auc:.4f}")
    
    # Check if this is the best model so far
    if auc > best_auc:
        best_auc = auc
        best_model_name = name
        best_model = pipeline

# Save the best model
print(f"\nBest model: {best_model_name} with AUC: {best_auc:.4f}")
joblib.dump(best_model, 'models/churn_model.pkl')
print("Best model saved to 'models/churn_model.pkl'")

# Save results summary
with open('models/model_comparison.txt', 'w') as f:
    f.write("Model Comparison Results\n")
    f.write("=======================\n\n")
    for name, metrics in results.items():
        f.write(f"{name}:\n")
        f.write(f"  Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"  Precision: {metrics['precision']:.4f}\n")
        f.write(f"  Recall: {metrics['recall']:.4f}\n")
        f.write(f"  AUC: {metrics['auc']:.4f}\n\n")
    f.write(f"Best model: {best_model_name} with AUC: {best_auc:.4f}\n")

print("\nModel training complete!")