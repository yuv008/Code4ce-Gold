import nltk
import pandas as pd
import numpy as np
from pymongo import MongoClient
import logging
from typing import List, Optional, Union
from datetime import datetime
from collections import Counter
import re
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from string import punctuation

class TextPreprocessor:
    """
    A class to handle text preprocessing operations for news articles.
    """
    def __init__(self, mongo_uri: str, database: str, collection: str):
        """
        Initialize the TextPreprocessor with MongoDB connection details and download required NLTK data.
        
        Args:
            mongo_uri (str): MongoDB connection URI
            database (str): Database name
            collection (str): Collection name
        """
        self.mongo_uri = mongo_uri
        self.database = database
        self.collection = collection
        self.setup_logging()
        self.download_nltk_resources()
        self.initialize_preprocessing_tools()

    def setup_logging(self) -> None:
        """Configure logging settings."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('text_preprocessing.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def download_nltk_resources(self) -> None:
        """Download required NLTK resources."""
        required_resources = [
            'punkt',
            'stopwords',
            'wordnet',
            'averaged_perceptron_tagger'
        ]
        
        for resource in required_resources:
            try:
                nltk.download(resource, quiet=True)
                self.logger.info(f"Successfully downloaded NLTK resource: {resource}")
            except Exception as e:
                self.logger.error(f"Failed to download NLTK resource {resource}: {str(e)}")

    def initialize_preprocessing_tools(self) -> None:
        """Initialize text preprocessing tools."""
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        self.stop_words.update(set(punctuation))
        
        # Add custom stop words if needed
        custom_stop_words = {'quot', 'amp', 'would', 'could', 'should', 'said'}
        self.stop_words.update(custom_stop_words)

    def connect_to_mongodb(self) -> Optional[MongoClient]:
        """
        Establish connection to MongoDB.
        
        Returns:
            Optional[MongoClient]: MongoDB client or None if connection fails
        """
        try:
            client = MongoClient(self.mongo_uri, serverSelectionTimeoutMS=5000)
            client.server_info()  # Verify connection
            return client
        except Exception as e:
            self.logger.error(f"MongoDB connection failed: {str(e)}")
            return None

    def clean_text(self, text: Union[str, float]) -> str:
        """
        Clean text by removing special characters and normalizing.
        
        Args:
            text: Input text to clean
            
        Returns:
            str: Cleaned text
        """
        if pd.isna(text):
            return ""
            
        text = str(text).lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text.strip()

    def tokenize_text(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text (str): Input text
            
        Returns:
            List[str]: List of tokens
        """
        try:
            return word_tokenize(text)
        except Exception as e:
            self.logger.warning(f"Tokenization failed: {str(e)}")
            return text.split()

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Remove stop words from token list.
        
        Args:
            tokens (List[str]): List of tokens
            
        Returns:
            List[str]: Filtered tokens
        """
        return [token for token in tokens if token.lower() not in self.stop_words]

    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """
        Lemmatize tokens.
        
        Args:
            tokens (List[str]): List of tokens
            
        Returns:
            List[str]: Lemmatized tokens
        """
        return [self.lemmatizer.lemmatize(token) for token in tokens]

    def stem_tokens(self, tokens: List[str]) -> List[str]:
        """
        Apply Porter stemming to tokens.
        
        Args:
            tokens (List[str]): List of tokens
            
        Returns:
            List[str]: Stemmed tokens
        """
        return [self.stemmer.stem(token) for token in tokens]

    def extract_text_features(self, text: str) -> dict:
        """
        Extract additional text features.
        
        Args:
            text (str): Input text
            
        Returns:
            dict: Dictionary of text features
        """
        sentences = sent_tokenize(text)
        words = word_tokenize(text.lower())
        
        return {
            'sentence_count': len(sentences),
            'word_count': len(words),
            'avg_word_length': np.mean([len(word) for word in words]) if words else 0,
            'avg_sentence_length': np.mean([len(word_tokenize(sent)) for sent in sentences]) if sentences else 0
        }

    def get_top_keywords(self, tokens: List[str], n: int = 5) -> List[tuple]:
        """
        Get top n keywords based on frequency.
        
        Args:
            tokens (List[str]): List of tokens
            n (int): Number of top keywords to return
            
        Returns:
            List[tuple]: List of (word, frequency) tuples
        """
        return Counter(tokens).most_common(n)

    def preprocess_text(self, df: pd.DataFrame, text_columns: List[str] = ['summary']) -> pd.DataFrame:
        """
        Apply comprehensive text preprocessing to specified columns.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            text_columns (List[str]): List of columns to process
            
        Returns:
            pd.DataFrame: Processed DataFrame
        """
        df = df.copy()
        processed_count = 0

        for column in text_columns:
            if column not in df.columns:
                self.logger.warning(f"Column '{column}' not found in DataFrame")
                continue

            self.logger.info(f"Processing column: {column}")
            
            # Basic text cleaning
            df[f'cleaned_{column}'] = df[column].apply(self.clean_text)
            
            # Tokenization
            df[f'tokens_{column}'] = df[f'cleaned_{column}'].apply(self.tokenize_text)
            
            # Stop words removal
            df[f'filtered_tokens_{column}'] = df[f'tokens_{column}'].apply(self.remove_stopwords)
            
            # Lemmatization
            df[f'lemmatized_{column}'] = df[f'filtered_tokens_{column}'].apply(self.lemmatize_tokens)
            
            # Stemming
            df[f'stemmed_{column}'] = df[f'filtered_tokens_{column}'].apply(self.stem_tokens)
            
            # Extract text features
            features = df[column].apply(self.extract_text_features)
            for feature_name, values in zip(
                ['sentence_count', 'word_count', 'avg_word_length', 'avg_sentence_length'],
                zip(*[d.values() for d in features])
            ):
                df[f'{column}_{feature_name}'] = list(values)
            
            # Get top keywords
            df[f'{column}_top_keywords'] = df[f'lemmatized_{column}'].apply(
                lambda x: self.get_top_keywords(x)
            )
            
            processed_count += 1

        # Add metadata
        df['preprocessed_at'] = datetime.now()
        
        self.logger.info(f"Processed {processed_count} columns")
        return df

    def process_and_save(self) -> bool:
        """
        Main method to load data, preprocess it, and save back to MongoDB.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Connect to MongoDB
            client = self.connect_to_mongodb()
            if not client:
                return False

            db = client[self.database]
            collection = db[self.collection]

            # Load data
            self.logger.info("Loading data from MongoDB...")
            articles = list(collection.find())
            if not articles:
                self.logger.error("No data found in MongoDB collection")
                return False

            df = pd.DataFrame(articles)
            initial_shape = df.shape
            self.logger.info(f"Loaded {initial_shape[0]} documents")

            # Preprocess text
            preprocessed_df = self.preprocess_text(df)
            
            # Save to new collection
            preprocessed_collection_name = f"preprocessed_{self.collection}"
            preprocessed_collection = db[preprocessed_collection_name]
            
            # Convert ObjectId to string for JSON serialization
            preprocessed_df['_id'] = preprocessed_df['_id'].astype(str)
            
            # Insert preprocessed data
            preprocessed_collection.insert_many(preprocessed_df.to_dict('records'))
            self.logger.info(f"Data saved to '{preprocessed_collection_name}'")
            
            return True

        except Exception as e:
            self.logger.error(f"Error in process_and_save: {str(e)}")
            return False
        
        finally:
            if 'client' in locals():
                client.close()

if __name__ == "__main__":
    # Configuration
    MONGO_URI = "mongodb://localhost:27017"
    DATABASE_NAME = "news_aggregator"
    COLLECTION_NAME = "cleaned_articles"

    # Initialize and run preprocessor
    preprocessor = TextPreprocessor(MONGO_URI, DATABASE_NAME, COLLECTION_NAME)
    success = preprocessor.process_and_save()
    
    if success:
        print("Text preprocessing completed successfully")
    else:
        print("Text preprocessing failed. Check logs for details")