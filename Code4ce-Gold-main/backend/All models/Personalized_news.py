import pandas as pd
import numpy as np
from pymongo import MongoClient
from bson.objectid import ObjectId
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from newspaper import Article
import nltk
from typing import Dict, List, Optional
from datetime import datetime

# MongoDB Connection Handler
class MongoDBHandler:
    def __init__(self, connection_string):
        """
        Initialize MongoDB connection with Atlas
        connection_string format:
        "mongodb+srv://<username>:<password>@<cluster-url>/<dbname>?retryWrites=true&w=majority"
        """
        self.client = MongoClient(connection_string)
        self.db = self.client.get_default_database()
        
        # Collections
        self.articles = self.db.articles
        self.users = self.db.users
        self.interactions = self.db.interactions
        
        # Create indexes
        self.articles.create_index("url", unique=True)
        self.articles.create_index("publish_date")
        self.interactions.create_index([("user_id", 1), ("article_id", 1)])
        self.interactions.create_index("timestamp")

# Text Analysis Components (remains same as before)
class TextAnalyzer:
    def __init__(self):
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )
        
        self.summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn"
        )
    
    def analyze_sentiment(self, text: str) -> Dict:
        result = self.sentiment_analyzer(text[:512])[0]
        return {
            'sentiment': result['label'],
            'score': result['score']
        }
    
    def generate_summary(self, text: str, max_length: int = 130) -> str:
        if len(text.split()) > 1024:
            chunks = self._split_text(text)
            summaries = []
            for chunk in chunks:
                summary = self.summarizer(
                    chunk,
                    max_length=max_length,
                    min_length=30,
                    do_sample=False
                )[0]['summary_text']
                summaries.append(summary)
            return " ".join(summaries)
        else:
            return self.summarizer(
                text,
                max_length=max_length,
                min_length=30,
                do_sample=False
            )[0]['summary_text']
    
    def _split_text(self, text: str) -> List[str]:
        sentences = nltk.sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            words = len(sentence.split())
            if current_length + words <= 1024:
                current_chunk.append(sentence)
                current_length += words
            else:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_length = words
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks

# Article Manager with MongoDB Integration
class ArticleManager:
    def __init__(self, mongo_handler: MongoDBHandler):
        self.text_analyzer = TextAnalyzer()
        self.mongo = mongo_handler
    
    def process_article(self, url: str) -> Optional[ObjectId]:
        # Check if article already exists
        existing_article = self.mongo.articles.find_one({"url": url})
        if existing_article:
            return existing_article["_id"]
        
        article = Article(url)
        try:
            article.download()
            article.parse()
            article.nlp()
            
            content = article.text
            custom_summary = self.text_analyzer.generate_summary(content)
            sentiment = self.text_analyzer.analyze_sentiment(content)
            
            article_data = {
                "title": article.title,
                "content": content,
                "summary": custom_summary,
                "newspaper_summary": article.summary,
                "url": url,
                "image_url": article.top_image,
                "keywords": article.keywords,
                "publish_date": article.publish_date or datetime.utcnow(),
                "sentiment": sentiment["sentiment"],
                "sentiment_score": sentiment["score"],
                "created_at": datetime.utcnow()
            }
            
            result = self.mongo.articles.insert_one(article_data)
            return result.inserted_id
            
        except Exception as e:
            print(f"Error processing article {url}: {str(e)}")
            return None

# Enhanced Recommendation System with MongoDB
class NewsRecommendationSystem:
    def __init__(self, mongo_connection_string: str):
        self.mongo = MongoDBHandler(mongo_connection_string)
        self.article_manager = ArticleManager(self.mongo)
        self.content_recommender = None
        self.collaborative_model = None
        
    def process_new_article(self, url: str) -> Optional[ObjectId]:
        """Process and store a new article"""
        return self.article_manager.process_article(url)
    
    def _prepare_training_data(self):
        """Prepare training data from MongoDB collections"""
        # Get all articles
        articles_cursor = self.mongo.articles.find(
            {},
            {
                "_id": 1,
                "title": 1,
                "content": 1,
                "sentiment_score": 1,
                "keywords": 1
            }
        )
        articles_df = pd.DataFrame(list(articles_cursor))
        
        # Get all interactions
        interactions_cursor = self.mongo.interactions.find(
            {},
            {
                "user_id": 1,
                "article_id": 1,
                "interaction_type": 1,
                "timestamp": 1
            }
        )
        interactions_df = pd.DataFrame(list(interactions_cursor))
        
        return articles_df, interactions_df
    
    def train(self):
        """Train the recommendation models"""
        articles_df, interactions_df = self._prepare_training_data()
        
        # Initialize and train content-based recommender
        self.content_recommender = ContentBasedRecommender()
        self.content_recommender.fit(articles_df)
        
        # Train collaborative model
        self._train_collaborative_model(interactions_df, articles_df)
    
    def get_recommendations(self, user_id: str, n_recommendations: int = 5) -> List[Dict]:
        """Get personalized recommendations for a user"""
        # Get user's recent interactions
        recent_interactions = list(self.mongo.interactions.find(
            {"user_id": user_id},
            {"article_id": 1, "timestamp": 1}
        ).sort("timestamp", -1).limit(3))
        
        # Get content-based recommendations
        content_based_recs = set()
        for interaction in recent_interactions:
            recs = self.content_recommender.get_recommendations(
                interaction["article_id"],
                n_recommendations=3
            )
            content_based_recs.update(recs)
        
        # Get collaborative recommendations
        collaborative_recs = self._get_collaborative_recommendations(
            user_id,
            n_recommendations
        )
        
        # Combine recommendations
        final_rec_ids = list(content_based_recs) + collaborative_recs
        final_rec_ids = list(dict.fromkeys(final_rec_ids))[:n_recommendations]
        
        # Fetch full article details
        recommended_articles = list(self.mongo.articles.find(
            {"_id": {"$in": final_rec_ids}},
            {
                "title": 1,
                "summary": 1,
                "url": 1,
                "image_url": 1,
                "sentiment": 1,
                "sentiment_score": 1,
                "publish_date": 1
            }
        ))
        
        return recommended_articles
    
    def record_interaction(self, user_id: str, article_id: ObjectId, interaction_type: str):
        """Record a user's interaction with an article"""
        interaction_data = {
            "user_id": user_id,
            "article_id": article_id,
            "interaction_type": interaction_type,
            "timestamp": datetime.utcnow()
        }
        self.mongo.interactions.insert_one(interaction_data)
    
    def get_user_history(self, user_id: str) -> List[Dict]:
        """Get a user's reading history"""
        pipeline = [
            {"$match": {"user_id": user_id}},
            {"$sort": {"timestamp": -1}},
            {"$lookup": {
                "from": "articles",
                "localField": "article_id",
                "foreignField": "_id",
                "as": "article"
            }},
            {"$unwind": "$article"},
            {"$project": {
                "title": "$article.title",
                "url": "$article.url",
                "interaction_type": 1,
                "timestamp": 1
            }}
        ]
        
        return list(self.mongo.interactions.aggregate(pipeline))

# Example usage
def main():
    # MongoDB Atlas connection string
    connection_string = "mongodb+srv://<username>:<password>@<cluster-url>/<dbname>?retryWrites=true&w=majority"
    
    # Initialize the recommendation system
    recommender = NewsRecommendationSystem(connection_string)
    
    # Process a new article
    article_id = recommender.process_new_article(
        "https://example.com/news-article"
    )
    
    # Train the recommendation models
    recommender.train()
    
    # Get recommendations for a user
    user_id = "user123"
    recommendations = recommender.get_recommendations(user_id)
    
    # Print recommendations
    for article in recommendations:
        print(f"\nTitle: {article['title']}")
        print(f"Summary: {article['summary']}")
        print(f"Sentiment: {article['sentiment']} (Score: {article['sentiment_score']:.2f})")
        print(f"URL: {article['url']}")
        print(f"Image: {article['image_url']}")
        print(f"Published: {article['publish_date']}")

if __name__ == "__main__":
    main()