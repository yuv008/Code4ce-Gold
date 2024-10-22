import pandas as pd
import numpy as np
import re
from pymongo import MongoClient
from datetime import datetime
import logging
from typing import Optional, Dict, Any, Tuple
from urllib.parse import urlparse

class DataCleaner:
    """
    A class to handle data cleaning operations for news articles.
    """
    def __init__(self, mongo_uri: str, database: str, collection: str):
        """
        Initialize the DataCleaner with MongoDB connection details.
        
        Args:
            mongo_uri (str): MongoDB connection URI
            database (str): Database name
            collection (str): Collection name
        """
        self.mongo_uri = mongo_uri
        self.database = database
        self.collection = collection
        self.setup_logging()

    def setup_logging(self) -> None:
        """Configure logging settings."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('data_cleaning.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def connect_to_mongodb(self) -> Optional[MongoClient]:
        """
        Establish connection to MongoDB.
        
        Returns:
            Optional[MongoClient]: MongoDB client or None if connection fails
        """
        try:
            client = MongoClient(self.mongo_uri, serverSelectionTimeoutMS=5000)
            # Verify connection
            client.server_info()
            return client
        except Exception as e:
            self.logger.error(f"MongoDB connection failed: {str(e)}")
            return None

    def clean_text(self, text: str) -> str:
        """
        Clean text by removing special characters and standardizing format.
        
        Args:
            text (str): Input text to clean
            
        Returns:
            str: Cleaned text
        """
        if pd.isna(text):
            return ""
        
        if not isinstance(text, str):
            return str(text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text.strip()

    def standardize_date(self, date_str: Any) -> Optional[datetime]:
        """
        Convert various date formats to standard datetime.
        
        Args:
            date_str: Input date in various formats
            
        Returns:
            Optional[datetime]: Standardized datetime object or None if invalid
        """
        if pd.isna(date_str):
            return None

        # If already a datetime, return as is
        if isinstance(date_str, datetime):
            return date_str

        date_formats = [
            "%Y-%m-%d",
            "%d/%m/%Y",
            "%m/%d/%Y",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
        ]

        for fmt in date_formats:
            try:
                return pd.to_datetime(date_str, format=fmt)
            except:
                continue
        
        try:
            return pd.to_datetime(date_str)
        except:
            self.logger.warning(f"Could not parse date: {date_str}")
            return None

    def validate_url(self, url: str) -> bool:
        """
        Validate URL format.
        
        Args:
            url (str): URL to validate
            
        Returns:
            bool: True if URL is valid, False otherwise
        """
        if pd.isna(url):
            return False
            
        try:
            result = urlparse(str(url))
            return all([result.scheme, result.netloc])
        except:
            return False

    def log_dropped_rows(self, df: pd.DataFrame, new_df: pd.DataFrame, step: str) -> None:
        """
        Log information about dropped rows at each step.
        
        Args:
            df (pd.DataFrame): Original DataFrame
            new_df (pd.DataFrame): DataFrame after cleaning step
            step (str): Name of the cleaning step
        """
        dropped_count = len(df) - len(new_df)
        if dropped_count > 0:
            self.logger.info(f"{step}: Dropped {dropped_count} rows")
            if dropped_count < 10:  # Only show examples for a small number of drops
                dropped_df = df[~df.index.isin(new_df.index)]
                self.logger.info(f"Example dropped rows: {dropped_df.head().to_dict('records')}")

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform comprehensive data cleaning on the DataFrame.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        initial_rows = len(df)
        self.logger.info(f"Starting data cleaning process with {initial_rows} rows")

        # Create a copy to avoid modifying the original
        df = df.copy()

        # Print sample of initial data
        self.logger.info(f"Sample of initial data:\n{df.head(1).to_dict('records')}")

        # Basic validation
        required_columns = ['title', 'link', 'summary', 'published_date']
        existing_columns = df.columns.tolist()
        self.logger.info(f"Existing columns: {existing_columns}")
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            self.logger.error(f"Missing required columns: {missing_columns}")
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Print column info and missing values
        self.logger.info("\nColumn info:")
        for col in df.columns:
            null_count = df[col].isna().sum()
            self.logger.info(f"{col}: {null_count} null values")

        # Clean text fields
        self.logger.info("Cleaning text fields...")
        df['title'] = df['title'].apply(self.clean_text)
        df['summary'] = df['summary'].apply(self.clean_text)
        
        # Fill empty text fields with placeholders
        df['title'] = df['title'].replace('', 'Untitled')
        df['summary'] = df['summary'].replace('', 'No summary available')

        # Validate URLs - just log invalid URLs but don't remove them
        self.logger.info("Validating URLs...")
        invalid_urls = df[~df['link'].apply(self.validate_url)]['link'].tolist()
        if invalid_urls:
            self.logger.warning(f"Found {len(invalid_urls)} invalid URLs: {invalid_urls[:5]}")

        # Standardize dates
        self.logger.info("Standardizing dates...")
        df['published_date'] = df['published_date'].apply(self.standardize_date)
        
        # For missing dates, use current date instead of dropping
        df['published_date'] = df['published_date'].fillna(datetime.now())

        # Remove duplicates
        self.logger.info("Removing duplicates...")
        before_dedup = len(df)
        df = df.drop_duplicates(subset=['link'], keep='first')
        self.log_dropped_rows(df, df, f"Duplicate removal: {before_dedup - len(df)} duplicates removed")
        
        # Add metadata
        df['cleaned_at'] = datetime.now()
        df['word_count'] = df['summary'].str.split().str.len()

        # Final validation
        final_rows = len(df)
        rows_removed = initial_rows - final_rows
        self.logger.info(f"Cleaning complete. Removed {rows_removed} rows ({rows_removed/initial_rows:.2%})")
        
        if final_rows == 0:
            self.logger.error("All rows were removed during cleaning!")
            self.logger.error("Please check the previous log messages for details about where rows were dropped.")
        else:
            self.logger.info(f"Final row count: {final_rows}")
            self.logger.info(f"Sample of cleaned data:\n{df.head(1).to_dict('records')}")

        return df

    def process_and_save(self) -> bool:
        """
        Main method to load data, clean it, and save back to MongoDB.
        
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
            
            # Clean data
            cleaned_df = self.clean_data(df)
            
            if cleaned_df.empty:
                self.logger.error("No data left after cleaning")
                return False

            # Save to new collection
            cleaned_collection_name = f"cleaned_{self.collection}"
            cleaned_collection = db[cleaned_collection_name]
            
            # Convert ObjectId to string for JSON serialization
            cleaned_df['_id'] = cleaned_df['_id'].astype(str)
            
            # Insert cleaned data
            cleaned_collection.insert_many(cleaned_df.to_dict('records'))
            self.logger.info(f"Data saved to '{cleaned_collection_name}'")
            
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
    COLLECTION_NAME = "articles"

    # Initialize and run cleaner
    cleaner = DataCleaner(MONGO_URI, DATABASE_NAME, COLLECTION_NAME)
    success = cleaner.process_and_save()
    
    if success:
        print("Data cleaning completed successfully")
    else:
        print("Data cleaning failed. Check logs for details")