from flask import Flask, render_template, request
import pickle
import re
import string

app = Flask(__name__)

# Load the model and vectorizer
model = pickle.load(open('spam_model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    return text

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = ''
    if request.method == 'POST':
        message = request.form['message']
        processed_message = preprocess_text(message)
        vectorized_message = vectorizer.transform([processed_message])
        result = model.predict(vectorized_message)
        prediction = 'Spam' if result[0] == 1 else 'Not Spam'
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
