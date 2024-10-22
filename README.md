# Warcast: AI-Based News Aggregator Platform

### Project by: Code4Ce (Winner at COEP Mindspark 24 NeuroHack)

---

## 📖 Overview

**Warcast** is an AI-powered news aggregator platform designed to provide personalized and real-time strategic news from various sources. By leveraging advanced machine learning models and NLP techniques, Warcast curates news feeds that adapt to user preferences, delivering updates on the most relevant global and geopolitical content.

---

## Key Features:

- **Personalized News Feed**: Uses a hybrid collaborative filtering model to recommend news articles based on user interactions (likes, bookmarks, etc.).
- **Sentiment Analysis**: Utilizes fine-tuned `BERT` and `DistilBERT` models along with `nltk` for comprehensive sentiment analysis, classifying news as Positive, Negative, or Neutral.
- **Text Summarization**: Implements Facebook's `BART` model for generating concise summaries (150 words) of lengthy news articles.
- **Multilingual Support**: The platform supports multiple languages, ensuring accessibility to a diverse audience.
- **Real-time News**: Aggregates news from 10+ sources, updated regularly to ensure fresh content.

---

## 🔧 Technologies Used

- **Python**: Back-end development.
- **BERT & DistilBERT**: For sentiment analysis.
- **Facebook BART**: For text summarization.
- **Selenium & BeautifulSoup4**: For web scraping news content.
- **MongoDB**: For database storage of articles and user data.
- **Machine Learning & Deep Learning**: Hybrid models for personalized news recommendations.
- **Flask**: Back-end API for serving news content.
- **HTML, CSS, JavaScript**: Front-end user interface.

---

## ⚙️ How It Works

1. **Data Aggregation**:  
   Warcast scrapes news articles from 10+ sources using `Selenium` and `BeautifulSoup4`, storing relevant data such as headlines, summaries, and article metadata in MongoDB.

2. **Sentiment Analysis**:  
   Each article is analyzed for sentiment using fine-tuned `BERT` and `DistilBERT` models, with `nltk` helping categorize the sentiment into Positive, Negative, or Neutral.

3. **Summarization**:  
   Long articles are summarized into concise 150-word versions using Facebook’s `BART` (large CNN model), allowing users to quickly grasp key points.

4. **Personalization**:  
   User activities (likes, bookmarks) are tracked, and a hybrid machine learning model recommends tailored news content for each user.

5. **Multilingual Support**:  
   Users can view news in their preferred languages, enhancing the global reach of Warcast.

---

## 🚀 Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yuv008/Code4ce-Gold.git
   cd Code4ce-Gold
## Set Up MongoDB
Ensure MongoDB is installed and running locally or on a cloud instance.

## Run the Application
```bash
python app.py
