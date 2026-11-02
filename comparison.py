import joblib
import pandas as pd
import __main__

from Vectorization.bow_model import clean_text, add_pos_tags
from Vectorization.word2vec_model import Word2VecTransformer


# Make the saved Word2Vec model recognize its transformer
__main__.Word2VecTransformer = Word2VecTransformer


# Load the trained models
bow_model = joblib.load("trained_bow_model.pkl")
tfidf_model = joblib.load("trained_tfidf_model.pkl")
word2vec_model = joblib.load("trained_word2vec_model.pkl")


# Load comparison test dataset
test_data = pd.read_csv("sentiment_comparison_test.csv")


# Store predictions
bow_predictions = []
tfidf_predictions = []
word2vec_predictions = []


# Print table header
print("\n" + "=" * 100)

print(
    "Sentence | Actual Sentiment | BoW Prediction | "
    "TF-IDF Prediction | Word2Vec Prediction"
)

print("=" * 100)


# Test every sentence
for index, row in test_data.iterrows():

    # Clean title and sentence
    clean_title = clean_text(row["title"])
    clean_sentence = clean_text(row["sentence"])


    # Add POS tags
    pos_title = add_pos_tags(clean_title)
    pos_sentence = add_pos_tags(clean_sentence)


    # Create input DataFrame
    input_data = pd.DataFrame({
        "pos_review_title": [pos_title],
        "pos_review_text": [pos_sentence]
    })


    # Get BoW prediction
    bow_prediction = bow_model.predict(input_data)[0]


    # Get TF-IDF prediction
    tfidf_prediction = tfidf_model.predict(input_data)[0]


    # Get Word2Vec prediction
    word2vec_prediction = word2vec_model.predict(input_data)[0]


    # Store predictions
    bow_predictions.append(bow_prediction)
    tfidf_predictions.append(tfidf_prediction)
    word2vec_predictions.append(word2vec_prediction)


    # Print one sentence comparison
    print(
        f"{row['sentence']} | "
        f"{row['actual_sentiment']} | "
        f"{bow_prediction} | "
        f"{tfidf_prediction} | "
        f"{word2vec_prediction}"
    )


# Calculate accuracy for each model
bow_accuracy = (
    sum(
        prediction == actual
        for prediction, actual
        in zip(
            bow_predictions,
            test_data["actual_sentiment"]
        )
    )
    / len(test_data)
)


tfidf_accuracy = (
    sum(
        prediction == actual
        for prediction, actual
        in zip(
            tfidf_predictions,
            test_data["actual_sentiment"]
        )
    )
    / len(test_data)
)


word2vec_accuracy = (
    sum(
        prediction == actual
        for prediction, actual
        in zip(
            word2vec_predictions,
            test_data["actual_sentiment"]
        )
    )
    / len(test_data)
)


# Display model accuracy
print("=" * 100)

print("\nModel Accuracy:")
print(f"BoW + Naive Bayes: {bow_accuracy:.2%}")
print(f"TF-IDF + Naive Bayes: {tfidf_accuracy:.2%}")
print(f"Word2Vec + Logistic Reg: {word2vec_accuracy:.2%}")


# I don't implement the Stemming and Lemmatization in this project, because, it doesn't help to improve model's accuracy score.
