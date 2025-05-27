# preclassifier.py

import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from joblib import dump
import os

def download_nltk_data():
    """
    Download necessary NLTK resources.
    """
    nltk_packages = [
        "punkt",
        "stopwords",
        "wordnet",
        "averaged_perceptron_tagger"
    ]
    for package in nltk_packages:
        nltk.download(package)

def load_and_prepare_data(benign_path, malicious_path):
    """
    Load the CSV files and prepare the data by labeling and concatenating.
    """
    # Load the datasets
    benign_df = pd.read_csv(benign_path)
    malicious_df = pd.read_csv(malicious_path)

    # Assign labels
    benign_df["label"] = "good"
    malicious_df["label"] = "bad"

    # Concatenate subject and body into a new column "text"
    benign_df["text"] = benign_df["subject"].fillna('') + " " + benign_df["body"].fillna('')
    malicious_df["text"] = malicious_df["subject"].fillna('') + " " + malicious_df["body"].fillna('')

    # Combine both datasets
    combined_df = pd.concat([benign_df, malicious_df], ignore_index=True)

    return combined_df[["text", "label"]]

def main():
    # Step 1: Download required NLTK data
    download_nltk_data()

    # Step 2: Load and prepare data
    data = load_and_prepare_data("benign_emails.csv", "malicious_emails.csv")

    # Step 3: TF-IDF Vectorization
    vectorizer = TfidfVectorizer(max_features=1000, stop_words="english")
    X = vectorizer.fit_transform(data["text"])
    y = data["label"]

    # Step 4: Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Step 5: Train classifier
    classifier = LogisticRegression(max_iter=1000, random_state=42)
    classifier.fit(X_train, y_train)

    # Step 6: Evaluate classifier
    y_pred = classifier.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Step 7: Save the classifier and vectorizer
    dump(classifier, "pretrained_classifier.joblib")
    dump(vectorizer, "pretrained_vectorizer.joblib")

if __name__ == "__main__":
    main()
