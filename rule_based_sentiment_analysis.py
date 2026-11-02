import re
import pandas as pd


# Create a simple positive and negative word list
positive_words = {
    "excellent", "amazing", "great", "good", "love", "loved",
    "happy", "satisfied", "perfect", "fantastic", "wonderful",
    "best", "easy", "reliable", "useful", "impressive",
    "comfortable", "recommend", "recommended", "smooth", "well"
}

negative_words = {
    "terrible", "bad", "poor", "hate", "hated", "disappointed",
    "disappointing", "worst", "awful", "useless", "slow",
    "difficult", "unreliable", "cheap", "broken", "defective",
    "frustrating", "uncomfortable", "regret", "problem"
}


# Clean the review text
def clean_text(text):

    if pd.isna(text):
        return ""

    text = str(text).lower()

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Keep only letters and spaces
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# Define the sentiment based on positive and negative scores
def get_sentiment(positive_score, negative_score):

    if positive_score > negative_score:
        return "positive"

    elif negative_score > positive_score:
        return "negative"

    else:
        return "neutral"


# Predict sentiment using word-based rules
def rule_based_sentiment(title, review):

    # Combine title and review
    text = title + " " + review

    # Clean the text
    text = clean_text(text)

    # Convert text into individual words
    words = text.split()

    # Count positive words
    positive_score = sum(
        1 for word in words
        if word in positive_words
    )

    # Count negative words
    negative_score = sum(
        1 for word in words
        if word in negative_words
    )

    # Get final sentiment
    sentiment = get_sentiment(
        positive_score,
        negative_score
    )

    return sentiment


# Test the rule-based model
def test_rule_based_model():

    # Load comparison test dataset
    data = pd.read_csv("sentiment_comparison_test.csv")

    print("Sentence | Actual Sentiment | "
        "Predicted Sentiment")

    correct_predictions = 0

    # Test every sentence
    for i in range(data.shape[0]):

        prediction = rule_based_sentiment(
            data["title"][i],
            data["sentence"][i]
        )

        print(
            f"{i + 1}. "
            f"{data['sentence'][i]} | "
            f"{data['actual_sentiment'][i]} | "
            f"{prediction}"
        )

        # Check whether prediction is correct
        if prediction == data["actual_sentiment"][i]:
            correct_predictions += 1

    # Calculate accuracy
    accuracy = (correct_predictions / data.shape[0])

    print()
    print("=" * 100)

    print(f"\nRule-Based Model Accuracy: "f"{accuracy:.2%}")


# Run testing only when this file is executed directly
if __name__ == "__main__":
    test_rule_based_model()

