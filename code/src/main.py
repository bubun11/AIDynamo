import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("emails.csv")

# Convert labels to numbers (Spam → 1, Ham → 0)
df['label'] = df['label'].map({"Adjustment": 1, "AUTransfer": 2, "ClosingNotice": 3, "CommitmentChange": 4, "FeePayment": 5, "MoneyMovementInbound": 6, "MoneyMovementOutbound": 7})

# Split dataset into training & testing sets
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)
# Initialize TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words="english")

# Transform text data into numerical form
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)
# Train the classifier
clf = MultinomialNB()
clf.fit(X_train_tfidf, y_train)

# Predict on test data
y_pred = clf.predict(X_test_tfidf)

# Evaluate model performance
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

def classify_email(email_text):
    email_tfidf = tfidf.transform([email_text])  # Convert text to TF-IDF
    prediction = clf.predict(email_tfidf)  # Predict label
    if prediction[0] == 1:
        return "Adjustment"
    elif prediction[0] == 2:
        return "AUTransfer"
    elif prediction[0] == 3:
        return "ClosingNotice"
    elif prediction[0] == 4:
        return "CommitmentChange"
    elif prediction[0] == 5:
        return "FeePayment"
    elif prediction[0] == 6:
        return "MoneyMovementInbound"
    elif prediction[0] == 7:
        return "MoneyMovementOutbound"
    else:
        return "Unknown"

# Test classification
new_email = "Hi it Adjustment 12!"
print("Prediction:", classify_email(new_email))
