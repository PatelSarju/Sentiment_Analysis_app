import re
import pandas as pd
import nltk
import numpy as np
import joblib

from gensim.models import Word2Vec

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


# Download required NLTK resources
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")


# Clean text
def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text)
    text = text.lower()

    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


# Add POS tags to the text
def add_pos_tags(text):
    words = nltk.word_tokenize(text)
    tagged_words = nltk.pos_tag(words)

    pos_text = " ".join(
        word + "_" + tag
        for word, tag in tagged_words
    )

    return pos_text


# Convert text into an average Word2Vec vector
def get_average_vector(text, model):

    words = text.split()

    vectors = []

    for word in words:
        if word in model.wv:
            vectors.append(model.wv[word])

    if len(vectors) == 0:
        return np.zeros(model.vector_size)

    return np.mean(vectors, axis=0)


# Custom transformer for Word2Vec
class Word2VecTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, word2vec_title, word2vec_text):
        self.word2vec_title = word2vec_title
        self.word2vec_text = word2vec_text

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        title_vectors = np.array([
            get_average_vector(
                text,
                self.word2vec_title
            )
            for text in X["pos_review_title"]
        ])

        text_vectors = np.array([
            get_average_vector(
                text,
                self.word2vec_text
            )
            for text in X["pos_review_text"]
        ])

        return np.hstack([
            title_vectors,
            text_vectors
        ])


def train_word2vec_model():

    # Load dataset
    df = pd.read_csv("customer_reviews_sentiment_practice.csv")

    # Keep only positive and negative reviews
    df = df[df["sentiment"].isin(["positive", "negative"])]

    # Clean title and review text
    df["clean_title"] = df["review_title"].apply(clean_text)
    df["clean_review"] = df["review_text"].apply(clean_text)

    # Add POS tags
    df["pos_review_title"] = df["clean_title"].apply(add_pos_tags)
    df["pos_review_text"] = df["clean_review"].apply(add_pos_tags)

    # Input and target
    X = df[["pos_review_title", "pos_review_text"]]
    y = df["sentiment"]

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


    # Prepare title sentences
    title_sentences = [
        text.split()
        for text in X_train["pos_review_title"]
    ]

    # Prepare review sentences
    text_sentences = [
        text.split()
        for text in X_train["pos_review_text"]
    ]


    # Train Word2Vec model for titles
    word2vec_title = Word2Vec(
        sentences=title_sentences,
        vector_size=150,
        window=5,
        min_count=1,
        workers=4,
        sg=1,
        epochs=30
    )


    # Train Word2Vec model for reviews
    word2vec_text = Word2Vec(
        sentences=text_sentences,
        vector_size=200,
        window=5,
        min_count=1,
        workers=4,
        sg=1,
        epochs=30
    )


    # Create Word2Vec transformer
    word2vec_transformer = Word2VecTransformer(
        word2vec_title=word2vec_title,
        word2vec_text=word2vec_text
    )


    # Create complete pipeline
    word2vec_model = Pipeline(
        steps=[
            ("word2vec", word2vec_transformer),

            (
                "classifier",
                LogisticRegression(
                    C=1,
                    max_iter=1000,
                    class_weight="balanced"
                )
            )
        ]
    )


    # Train the model
    word2vec_model.fit(
        X_train,
        y_train
    )


    # Make predictions
    y_pred_word2vec = word2vec_model.predict(X_test)


    # Evaluation
    # print("Word2Vec + Logistic Regression")

    # print(
    #     classification_report(
    #         y_test,
    #         y_pred_word2vec
    #     )
    # )

    # print("Confusion Matrix:")

    # print(
    #     confusion_matrix(
    #         y_test,
    #         y_pred_word2vec
    #     )
    # )


    # Save trained model
    joblib.dump(
        word2vec_model,
        "trained_word2vec_model.pkl"
    )

    print("\nWord2Vec model saved successfully.")


if __name__ == "__main__":
    train_word2vec_model()
    
    
# I use skip gram approach in word2vec model, because, when i use CBOW approach, it drop the accuracy score by 20%