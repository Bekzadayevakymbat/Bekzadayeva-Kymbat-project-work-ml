import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
    average_precision_score
)
import joblib


# Path to the phishing URL dataset
DATA_PATH = r"C:\Users\Huawei\Documents\project\phishing_site_urls.csv\phishing_site_urls.csv"


# Load the dataset from CSV file into a DataFrame
df = pd.read_csv(DATA_PATH)


# Ensure consistent naming of the label column
# Some datasets use "Label", others use "label"
if "Label" in df.columns and "label" not in df.columns:
    df = df.rename(columns={"Label": "label"})


# Remove rows with missing values and duplicate URLs
df = df.dropna(subset=["URL", "label"]).drop_duplicates(subset=["URL"])


# Convert all URLs to string type
df["URL"] = df["URL"].astype(str)


# Convert textual labels into binary numerical labels
# 1 represents phishing, 0 represents legitimate
def encode_label(x):
    x = str(x).strip().lower()
    return 1 if x in ["bad", "phishing", "malicious", "defacement", "1"] else 0


# Apply label encoding to the dataset
df["label"] = df["label"].apply(encode_label).astype(int)


# Split the dataset into training and testing sets
# Stratification preserves the class distribution
X_train, X_test, y_train, y_test = train_test_split(
    df["URL"],
    df["label"].values,
    test_size=0.2,
    random_state=42,
    stratify=df["label"].values
)


# Initialize TF-IDF vectorizer using character-level n-grams
# Character n-grams are effective for detecting phishing patterns in URLs
vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 6),
    min_df=2,
    max_features=300000
)


# Fit the vectorizer on training data and transform both sets
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


# Create a Linear Support Vector Machine classifier
# Class balancing compensates for class imbalance in phishing datasets
base = LinearSVC(class_weight="balanced")


# Calibrate the SVM classifier to enable probability estimation
model = CalibratedClassifierCV(
    base,
    method="sigmoid",
    cv=3
)


# Train the calibrated model on the training data
model.fit(X_train_vec, y_train)


# Predict phishing probabilities for the test set
proba = model.predict_proba(X_test_vec)[:, 1]


# Convert probabilities into binary predictions using threshold 0.5
pred = (proba >= 0.5).astype(int)


# Print evaluation metrics
print("Accuracy:", accuracy_score(y_test, pred))
print("ROC-AUC:", roc_auc_score(y_test, proba))
print("PR-AUC:", average_precision_score(y_test, proba))
print(classification_report(y_test, pred, digits=4))
print(confusion_matrix(y_test, pred))


# Save the trained model and TF-IDF vectorizer to disk
# These files are later used in the web application
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")


print("Saved: model.pkl")
print("Saved: vectorizer.pkl")
