import tkinter as tk
from tkinter import messagebox
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Preprocessing
import nltk

nltk.download('wordnet')
ps = WordNetLemmatizer()
stopwords_list = stopwords.words('english')


def clean_row(row):
    row = row.lower()
    row = re.sub('[^a-zA-Z]', ' ', row)
    tokens = row.split()
    cleaned_news = [ps.lemmatize(word) for word in tokens if word not in stopwords_list]
    return ' '.join(cleaned_news)


# Sample trained data for demonstration purposes
sample_data = ["This is a sample text about news", "This text is about fake news"]

# Initializing vectorizer and classifier
vectorizer = TfidfVectorizer(max_features=50000, lowercase=False, ngram_range=(1, 2))
vectorizer.fit(sample_data)  # Fit with your actual data

# Placeholder for actual model training
clf = MultinomialNB()


# clf.fit(sample_train_data, sample_train_labels) # Fit with your actual train data and labels

# Function to make predictions
def predict_news():
    input_text = news_entry.get("1.0", "end-1c")  # Get input from Text widget
    if not input_text.strip():
        messagebox.showwarning("Input Error", "Please enter some news text!")
        return

    # Preprocess and predict
    cleaned_news = clean_row(input_text)
    vectorized_news = vectorizer.transform([cleaned_news]).toarray()
    # Prediction - ensure model is trained before use
    prediction = clf.predict(vectorized_news)

    if prediction == 0:
        result_label.config(text="Prediction: News is True", fg="green")
    else:
        result_label.config(text="Prediction: News is Fake", fg="red")


# Setting up the Tkinter GUI
window = tk.Tk()
window.title("Spam News Detection")
window.geometry("600x400")
window.resizable(False, False)

# Title Label
title_label = tk.Label(window, text="Spam News Detection", font=("Helvetica", 16, "bold"))
title_label.pack(pady=10)

# Input Label
input_label = tk.Label(window, text="Enter News Text:", font=("Helvetica", 12))
input_label.pack(pady=5)

# Text Entry Widget
news_entry = tk.Text(window, wrap="word", width=70, height=10)
news_entry.pack(pady=10)

# Predict Button
predict_button = tk.Button(window, text="Predict", font=("Helvetica", 12), command=predict_news)
predict_button.pack(pady=10)

# Result Label
result_label = tk.Label(window, text="Prediction: ", font=("Helvetica", 14))
result_label.pack(pady=20)

# Start the Tkinter event loop
window.mainloop()
