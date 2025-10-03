import nltk

# Download necessary resources
nltk.download('movie_reviews')
nltk.download('punkt')
nltk.download('stopwords')
#loads reviews and suffles them so no biases
from nltk.corpus import movie_reviews
import random

# Load the dataset
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

# Shuffle to mix positive/negative reviews
random.shuffle(documents)

print(f"Loaded {len(documents)} reviews")
#cleans review and conerts to single string
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

data = []
labels = []

# Convert word lists to clean text
for review, label in documents:
    # Keep only alphabet letters and lowercase
    words = [w.lower() for w in review if w.isalpha()]
    # Remove stopwords
    words = [w for w in words if w not in stop_words]
    # Join back to sentence
    data.append(" ".join(words))
    labels.append(label)

print(f"Sample cleaned review:\n{data[0]}\nLabel: {labels[0]}")
#to convert text to numbers so model can understand
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Convert text to TF-IDF vectors
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(data)

# Encode labels: pos → 1, neg → 0
y = [1 if label == 'pos' else 0 for label in labels]

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Data converted to numbers and split into train/test")

#to train model and predicts and shows how accurate it is
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

#saves the model as pkl file
import joblib

# Save the trained model and the vectorizer
joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model and vectorizer saved successfully.")

