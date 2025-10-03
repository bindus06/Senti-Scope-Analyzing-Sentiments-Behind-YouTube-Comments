from flask import Flask, request, render_template
import joblib
import numpy as np

# Load the model and vectorizer
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Initialize Flask app
app = Flask(__name__)

# Route for home page (index)
@app.route('/')
def home():
    return render_template('index.html')

# Route to predict sentiment
@app.route('/predict', methods=['POST'])
def predict():
    review = request.form['review']  # Get the review from the form
    review_vectorized = vectorizer.transform([review])  # Transform the review
    prediction = model.predict(review_vectorized)  # Predict sentiment

    # Return the result
    if prediction == 1:
        result = "Positive"
    else:
        result = "Negative"
    return render_template('index.html', prediction_text='Sentiment: {}'.format(result))

if __name__ == '__main__':
    app.run(debug=True)
