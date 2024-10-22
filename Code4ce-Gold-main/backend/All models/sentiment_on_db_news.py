from datetime import datetime
import os
import nltk
import logging
from typing import Any, Dict, List
from transformers import pipeline
from nltk.tokenize import sent_tokenize
import numpy as np
from pathlib import Path
from pymongo import MongoClient  # Import MongoDB client
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
class SentimentAnalyzer:

    def _initialize_nltk(self):
        """
        Initialize NLTK resources with proper error handling and complete resource downloads.
        Downloads all required NLTK data to a specified directory.
        """
        try:
            import nltk
            
            # Create NLTK data directory if it doesn't exist
            nltk_data_dir = os.path.expanduser('~/nltk_data')
            os.makedirs(nltk_data_dir, exist_ok=True)
            
            # Set NLTK data path
            nltk.data.path.append(nltk_data_dir)
            
            # List of required NLTK resources
            required_resources = [
                'punkt',
                'averaged_perceptron_tagger',
                'maxent_ne_chunker',
                'words',
                'tokenizers/punkt'
            ]
            
            # Download each required resource
            for resource in required_resources:
                try:
                    nltk.download(resource, quiet=True, raise_on_error=True)
                    self.logger.info(f"Successfully downloaded NLTK resource: {resource}")
                except Exception as e:
                    self.logger.warning(f"Error downloading {resource}: {str(e)}")
                    # Try alternative download method
                    try:
                        nltk.download(resource, download_dir=nltk_data_dir)
                        self.logger.info(f"Successfully downloaded {resource} to {nltk_data_dir}")
                    except Exception as e2:
                        self.logger.error(f"Failed to download {resource} after retry: {str(e2)}")
                        raise
            
            # Verify downloads
            for resource in required_resources:
                if not nltk.data.find(resource, paths=[nltk_data_dir]):
                    raise LookupError(f"Resource {resource} not found after download attempt")
            
            self.logger.info("NLTK initialization completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing NLTK resources: {str(e)}")
            raise


    def __init__(self):
        """
        Initialize the sentiment analyzer with required NLTK resources and models.
        Also set up MongoDB connection.
        """
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize NLTK resources
        self.logger.info("Initializing NLTK resources...")
        self._initialize_nltk()
        
        
        # Initialize the transformer sentiment analysis pipeline
        try:
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
            )
        except Exception as e:
            self.logger.error(f"Error initializing sentiment pipeline: {str(e)}")
            raise

    def _initialize_nltk(self):
        """Initialize NLTK resources with proper error handling."""
        try:
            nltk_data_dir = Path.home() / 'nltk_data'
            nltk_data_dir.mkdir(parents=True, exist_ok=True)
            
            # Download required NLTK data
            nltk.download('punkt', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            
        except Exception as e:
            self.logger.error(f"Error initializing NLTK resources: {str(e)}")
            raise

    # def fetch_news_from_db(self, mongo_uri: str, db_name: str, collection_name: str) -> List[Dict]:
    #     """
    #     Fetch news articles from MongoDB.
        
    #     Args:
    #         mongo_uri (str): MongoDB connection string.
    #         db_name (str): Name of the database.
    #         collection_name (str): Collection name from where news will be fetched.
            
    #     Returns:
    #         List[Dict]: List of news articles with their content and timestamps.
    #     """
    #     try:
    #         # Connect to MongoDB
    #         client = MongoClient(mongo_uri)
    #         db = client[db_name]
    #         collection = db[collection_name]
            
    #         # Fetch news articles from MongoDB (content and timestamp only)
    #         news_articles = collection.find({}, {'summary': 1, 'published_date': 1})
            
    #         # Return the fetched articles as a list
    #         return list(news_articles)
            
    #     except Exception as e:
    #         self.logger.error(f"Error fetching news from MongoDB: {str(e)}")
    #         return []
    def fetch_news_from_db(self, mongo_uri: str, db_name: str, collection_name: str) -> List[Dict]:
        """
        Fetch news articles' summaries from MongoDB with proper projection and filtering.
        
        Args:
            mongo_uri (str): MongoDB connection string.
            db_name (str): Name of the database.
            collection_name (str): Collection name from where news will be fetched.
            start_date (datetime, optional): Start date for filtering articles.
            end_date (datetime, optional): End date for filtering articles.
            
        Returns:
            List[Dict]: List of news articles with their summaries and timestamps.
        """
        try:
            # Connect to MongoDB
            client = MongoClient(mongo_uri)
            db = client[db_name]
            collection = db[collection_name]
            
            # Build query for date filtering
            query = {}
            
            # Fetch only the summary and published_date fields with proper projection
            projection = {
                '_id':1,
                'summary': 1,
                'published_date': 1,
                '_id': 0  # Exclude _id field
            }
            
            # Execute query with projection
            news_articles = collection.find(
                query,
                projection
            ).sort('published_date', -1)
            
            # Process results
            processed_articles = []
            for article in news_articles:
                if 'summary' in article and article['summary']:
                    id = article['id']
                    # Clean the summary text
                    summary = str(article['summary'])
                    # Remove any null bytes that might cause issues
                    summary = summary.replace('\x00', '')
                    # Strip whitespace
                    summary = summary.strip()
                    
                    if summary:  # Only include articles with non-empty summaries
                        processed_articles.append({
                          
                            'summary': summary,
                            'published_date': article.get('published_date')
                        })
            
            return processed_articles
                
        except Exception as e:
            self.logger.error(f"Error fetching news from MongoDB: {str(e)}")
            return []
        finally:
            if 'client' in locals():
                client.close()

    def analyze_text(self, text: str) -> Dict:
        """
        Analyze sentiment of a single text.
        
        Args:
            text (str): Input text for sentiment analysis.
            
        Returns:
            Dict: Sentiment analysis results.
        """
        if not text or not isinstance(text, str):
            return {
                'score': 0,
                'magnitude': 0,
                'label': 'neutral',
                'confidence': 0,
                'sentences': []
            }
        
        try:
            # Tokenize text into sentences
            sentences = sent_tokenize(text)
            sentence_results = []
            overall_score = 0
            
            # Analyze sentiment for each sentence
            for sentence in sentences:
                if sentence.strip():
                    result = self.sentiment_pipeline(sentence)[0]
                    score = result['score'] if result['label'] == 'POSITIVE' else -result['score']
                    
                    sentence_results.append({
                        'text': sentence,
                        'sentiment': result['label'],
                        'confidence': result['score'],
                        'score': score
                    })
                    overall_score += score
            
            # Calculate overall sentiment metrics
            num_sentences = len(sentence_results)
            if num_sentences > 0:
                average_score = overall_score / num_sentences
                magnitude = sum(abs(s['score']) for s in sentence_results) / num_sentences
                label = 'neutral' if abs(average_score) < 0.1 else 'positive' if average_score > 0 else 'negative'
                confidence = np.mean([s['confidence'] for s in sentence_results])
            else:
                average_score, magnitude, label, confidence = 0, 0, 'neutral', 0
            
            return {
                'score': average_score,
                'magnitude': magnitude,
                'label': label,
                'confidence': confidence,
                'sentences': sentence_results
            }
            
        except Exception as e:
            self.logger.error(f"Error in sentiment analysis: {str(e)}")
            return {
                'score': 0,
                'magnitude': 0,
                'label': 'neutral',
                'confidence': 0,
                'sentences': [],
                'error': str(e)
            }

    def analyze_news(self, mongo_uri: str, db_name: str, collection_name: str) -> List[Dict]:
        """
        Fetch news articles from MongoDB and perform sentiment analysis on the content.
        
        Args:
            mongo_uri (str): MongoDB connection string.
            db_name (str): Name of the database.
            collection_name (str): Collection name from where news will be fetched.
            
        Returns:
            List[Dict]: List of sentiment analysis results for each news article.
        """
        # Fetch news articles from MongoDB
        news_articles = self.fetch_news_from_db(mongo_uri, db_name, collection_name)
        print(news_articles)
        sentiment_results = []
        
        # Perform sentiment analysis on each article
        for article in news_articles:
            content = article.get('summary','')
            print(content)
            timestamp = article.get('published_date', None)
            
            # Analyze the content sentiment
            sentiment_result = self.analyze_text(content)
            sentiment_result['timestamp'] = timestamp
            
            sentiment_results.append(sentiment_result)
            print(sentiment_result)
        return sentiment_results

    def save_sentiment(self, article_id: str, sentiment_data: Dict[str, Any],connection_string,database_name,collection_name) -> str:
        """
        Save sentiment analysis results for a specific content.
        
        Args:
            content_id: Unique identifier for the analyzed content
            sentiment_data: Dictionary containing sentiment analysis results
            
        Returns:
            str: ID of the inserted document
        """

        self.client = MongoClient(connection_string)
        self.db = self.client[database_name]
        self.collection = self.db[collection_name]
        
        # Create index for better query performance
        self.collection.create_index("content_id")

        document = {
            "content_id": article_id,
            "timestamp": datetime.utcnow(),
            "sentiment_scores": {
                "score": sentiment_data.get("score", 0),
                "magnitude": sentiment_data.get("magnitude", 0),
                "label":sentiment_data.get("label",0),
                "confidence":sentiment_data.get('confidence',0),
            },
     
        }
        
        result = self.collection.insert_one(document)
        return str(result.inserted_id)


def main():
    """Main execution function to perform sentiment analysis on news articles from MongoDB."""
    try:
        analyzer = SentimentAnalyzer()
        
        # MongoDB connection details
        mongo_uri = "mongodb://localhost:27017/"
        db_name = 'news_aggregator'
        collection_name = 'articles'
        
        # Perform sentiment analysis on news articles from MongoDB
        sentiment_results = analyzer.analyze_news(mongo_uri, db_name, collection_name)
        
        # Display the results
        for result in sentiment_results:
            analyzer.save_sentiment(result['id'],result,mongo_uri,db_name,"article_sentiments")
            print("Sentiment analysis result:", result)
        
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()
