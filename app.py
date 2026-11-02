import streamlit as st
import pandas as pd
import __main__

from Vectorization.bow_model import clean_text, add_pos_tags
from rule_based_sentiment_analysis import rule_based_sentiment


# Streamlit page settings
st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="💬",
    layout="centered"
)


# App title
st.title("Sentiment Analysis")

st.write(
    "Enter a review title and review text to compare "
    "different sentiment analysis approaches."
)


# User input
title = st.text_input(
    "Review Title",
    placeholder="Enter review title..."
)

review = st.text_area(
    "Review",
    placeholder="Enter your review..."
)


# Function to display sentiment with emoji
def show_sentiment(sentiment):

    if sentiment == "positive":
        return "Positive 😊"

    elif sentiment == "negative":
        return "Negative 😡"

    else:
        return "Neutral 😐"


# Predict button
if st.button("Analyze Sentiment"):

    if title.strip() == "" and review.strip() == "":
        st.warning("Please enter a title or review.")

    else:

        # Clean user input
        clean_title = clean_text(title)
        clean_review = clean_text(review)


        # Add POS tags
        pos_title = add_pos_tags(clean_title)
        pos_review = add_pos_tags(clean_review)


        # Create input DataFrame
        input_data = pd.DataFrame({
            "pos_review_title": [pos_title],
            "pos_review_text": [pos_review]
        })


        # Get rule-based prediction
        rule_prediction = rule_based_sentiment(
            title,
            review
        )


        # Display results
        st.subheader("Sentiment Predictions")


        st.write(
            f"**Sentiment is:** "
            f"{show_sentiment(rule_prediction)}"
        )
    
