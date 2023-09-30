import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib

# Load your labeled dataset (assuming it's in a CSV format with 'text' and 'label' columns)
data = pd.read_csv('testdata.csv')

# Split the data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Create a bag-of-words vectorizer
vectorizer = CountVectorizer()

# Transform the text data into numerical features
X_train = vectorizer.fit_transform(train_data['text'])
X_test = vectorizer.transform(test_data['text'])

# Create and train a Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train, train_data['label'])

# Make predictions on the test data
predictions = classifier.predict(X_test)

# Evaluate the model's accuracy
accuracy = accuracy_score(test_data['label'], predictions)
print(f'Accuracy: {accuracy:.2f}')

# Save the trained model and vectorizer
joblib.dump(classifier, 'sentiment_classifier.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
