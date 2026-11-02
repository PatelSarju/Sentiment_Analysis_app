import re
import pandas as pd
import nltk
import joblib

from nltk import pos_tag
from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix


# Download required NLTK resources

nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")


# Create text cleaning function
def clean_text(text):

    # Handle missing text
    if pd.isna(text):
        return ""

    # Convert text to string
    text = str(text)

    # Convert text to lowercase
    text = text.lower()

    # Remove HTML tags
    text = re.sub(r"<[^>]+>"," ",text)

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+"," ",text)

    # Keep letters and spaces
    # Words such as "not" and "never" are preserved
    text = re.sub(r"[^a-zA-Z\s]"," ",text)

    # Remove extra spaces
    text = re.sub(r"\s+"," ",text).strip()

    return text


# Create a POS tagging function
def add_pos_tags(text):

    # Convert the sentence into individual words
    words = word_tokenize(text)

    # Assign a POS tag to each word
    tagged_words = pos_tag(words)

    # Combine each word with its POS tag
    # Example: good_JJ product_NN
    pos_text = " ".join(
        word + "_" + tag
        for word, tag in tagged_words
    )

    return pos_text


# Create a function to train and save the BoW model
def train_bow_model():

    # load the dataset
    df = pd.read_csv("customer_reviews_sentiment_practice.csv")

    # filter only positive and negative class
    df = df[df["sentiment"].isin(["positive", "negative"])]


    # Clean the review text
    df["clean_review_text"] = df["review_text"].apply(clean_text)

    # Clean the review title
    df["clean_review_title"] = df["review_title"].apply(clean_text)


    # Add POS tags to the cleaned text
    df["pos_review_text"] = df["clean_review_text"].apply(add_pos_tags)

    # Add POS tags to the cleaned review title
    df["pos_review_title"] = df["clean_review_title"].apply(add_pos_tags)


    # Select input and target
    X = df[["pos_review_title", "pos_review_text"]]
    y = df["sentiment"]


    # split the data into training and testing set
    x_train, x_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


    # create bag of words vectorizer for review title
    bow_title = CountVectorizer(
        max_features=100,
        ngram_range=(1, 2)
    )


    # create bag of words vectorizer for review text
    bow = CountVectorizer(
        max_features=165,
        ngram_range=(1, 2)
    )


    # Apply BoW to both columns
    bow_preprocessor = ColumnTransformer(
        transformers=[
            ("title", bow_title, "pos_review_title"),
            ("text", bow, "pos_review_text")
        ]
    )


    # Create a pipeline for preprocessing and classification
    bow_model = Pipeline([
        ("preprocessor", bow_preprocessor),
        ("classifier", MultinomialNB())
    ])


    # Train the model
    bow_model.fit(x_train, y_train)


    # Predict sentiment for test reviews
    y_pred_bow = bow_model.predict(x_test)


    # Evaluate the model
    # print(classification_report(y_test, y_pred_bow))


    # Display the confusion matrix
    # print("\nConfusion Matrix:")
    # print(confusion_matrix(y_test,y_pred_bow))


    # Save the trained model
    joblib.dump(bow_model, "trained_bow_model.pkl")

    print("BoW model trained and saved successfully.")


# Train and save the model
if __name__ == "__main__":
    train_bow_model()

