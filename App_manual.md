# 💬 Sentiment Analysis

A simple **Streamlit NLP application** that predicts the sentiment of a customer review using a rule-based approach.

## ✨ Features

- 📝 Enter a review title and review text.
- 🧹 Clean the review text.
- 🔤 Analyze sentiment using predefined positive and negative words.
- 😊 Display the predicted sentiment with an emoji:
  - 😊 Positive
  - 😡 Negative
  - 😐 Neutral

## 🛠️ Technologies Used

- 🐍 Python
- 🎈 Streamlit
- 🐼 Pandas
- 🔍 Regular Expressions

## ⚙️ How It Works

1. 📝 User enters a review.
2. 🧹 The review text is cleaned.
3. 🔢 Positive and negative words are counted.
4. ⚖️ The scores are compared.
5. 🎯 The application displays the final sentiment.

## 📁 Project Structure

```
Sentiment Analysis/
│
├── app.py
├── rule_based_sentiment_analysis.py
```


## 🚀 How to Run

### 📦 Install the required libraries

- pip install streamlit pandas

### ▶️ Run the application

- python -m streamlit run app.py

## 🧪 Example
### 📝 Review:
- This product is excellent and very reliable.

## 🎯 Result:
- Sentiment is: Positive 😊

## 🎓 Purpose
- This project demonstrates a simple rule-based approach to customer sentiment analysis using a Streamlit interface.