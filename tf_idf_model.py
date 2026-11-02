import re
import pandas as pd
import nltk
import joblib

from nltk import pos_tag
from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
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
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Keep letters and spaces
    # Words such as "not" and "never" are preserved
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

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


# Create TF-IDF model and save it
def train_tfidf_model():

    # Load the dataset
    df = pd.read_csv("customer_reviews_sentiment_practice.csv")

    # Filter only positive and negative classes
    df = df[df["sentiment"].isin(["positive", "negative"])]


    # Clean the review text
    df["clean_review_text"] = df["review_text"].apply(clean_text)

    # Clean the review title
    df["clean_review_title"] = df["review_title"].apply(clean_text)


    # Add POS tags to review text
    df["pos_review_text"] = df["clean_review_text"].apply(add_pos_tags)

    # Add POS tags to review title
    df["pos_review_title"] = df["clean_review_title"].apply(add_pos_tags)


    # Select input and target
    X = df[["pos_review_title", "pos_review_text"]]
    y = df["sentiment"]


    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


    # Create TF-IDF vectorizer for review title
    tf_idf_title = TfidfVectorizer(
        max_features=100,
        ngram_range=(1, 2)
    )


    # Create TF-IDF vectorizer for review text
    tf_idf_text = TfidfVectorizer(
        max_features=160,
        ngram_range=(1, 2)
    )


    # Apply TF-IDF to both text columns
    tf_idf_preprocessor = ColumnTransformer(
        transformers=[
            ("title", tf_idf_title, "pos_review_title"),
            ("text", tf_idf_text, "pos_review_text")
        ]
    )


    # Create a pipeline for preprocessing and classification
    tf_idf_model = Pipeline([
        ("preprocessor", tf_idf_preprocessor),
        ("classifier", MultinomialNB())
    ])


    # Train the model
    tf_idf_model.fit(x_train, y_train)


    # Predict sentiment for test reviews
    y_pred_tf_idf = tf_idf_model.predict(x_test)


    # Evaluate the model
    # print(classification_report(y_test, y_pred_tf_idf))


    # Display the confusion matrix
    # print("\nConfusion Matrix:")
    # print(confusion_matrix(y_test, y_pred_tf_idf))


    # Save the trained model
    joblib.dump(tf_idf_model, "trained_tfidf_model.pkl")

    print("\nTF-IDF model trained and saved successfully.")


# Train and save the model
if __name__ == "__main__":
    train_tfidf_model()
    
