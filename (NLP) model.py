# Building a natural language processing (NLP) model for sentiment analysis
# pip install pandas scikit-learn

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Sample dataset for sentiment analysis
data = {
    'text': [
        'I love this product, it is amazing!',
        'This is the worst purchase I have ever made.',
        'I am very happy with the service.',
        'I will never buy from this store again.',
        'The quality is excellent!',
        'Terrible experience, very disappointed.',
        'Absolutely fantastic! Highly recommend it.',
        'Not good at all, I want a refund.',
        'Very satisfied with my order.',
        'The item broke after one use, awful.'
    ],
    'label': ['positive', 'negative', 'positive', 'negative', 'positive', 'negative', 'positive', 'negative', 'positive', 'negative']
}

# Convert the dataset to a DataFrame
df = pd.DataFrame(data)

# Stratify the split to ensure balanced class distribution in train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)

# Convert text data to feature vectors using CountVectorizer
vectorizer = CountVectorizer()
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

# Train a Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train_vectors, y_train)

# Predict on the test set
y_pred = model.predict(X_test_vectors)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print('Classification Report:')
print(classification_report(y_test, y_pred, zero_division=0))

# Test with custom reviews
custom_reviews = [
    "This product exceeded my expectations!",
    "Worst service ever, completely dissatisfied.",
    "The experience was okay, not great, not terrible.",
    "Highly recommend this to everyone!"
]
custom_review_vectors = vectorizer.transform(custom_reviews)
custom_predictions = model.predict(custom_review_vectors)

# Display predictions for custom reviews
for review, sentiment in zip(custom_reviews, custom_predictions):
    print(f"Review: \"{review}\" -> Sentiment: {sentiment}")
