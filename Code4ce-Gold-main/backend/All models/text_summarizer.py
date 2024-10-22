import os
import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Optional, Any, Union
from dataclasses import dataclass
import time
from concurrent.futures import ThreadPoolExecutor

from pymongo import MongoClient, UpdateOne
from pymongo.errors import PyMongoError
from transformers import pipeline
from transformers.tokenization_utils_base import TruncationStrategy
import numpy as np
from tqdm import tqdm

@dataclass
class SummaryConfig:
    """Configuration for summarization parameters"""
    max_length: int = 600
    min_length: int = 50
    do_sample: bool = False
    num_beams: int = 4
    temperature: float = 1.0
    batch_size: int = 8
    retry_attempts: int = 3
    chunk_size: int = 1000

class SummarizationError(Exception):
    """Custom exception for summarization errors"""
    pass

class MongoDBHandler:
    """Handles all MongoDB operations"""
    
    def __init__(self, uri: str, db_name: str, collection_name: str):
        self.client = MongoClient(uri)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]
        self.logger = logging.getLogger(__name__)

    def get_unsummarized_articles(self, 
                                batch_size: int,
                                query_filter: Optional[Dict] = None) -> List[Dict]:
        """Fetch articles that need summarization"""
        # default_filter = {
        #     "content": {"$exists": True, "$ne": ""},
        #     "$or": [
        #         {"summary": {"$exists": True}},
        #         {"summary": ""},
        #         {"summary_status": "failed"}
        #     ]
        # }
        # query_filter = query_filter or default_filter
        
        # return list(self.collection.find(query_filter).limit(batch_size))
        return list(self.collection.find().limit(batch_size))

    def bulk_update_summaries(self, updates: List[UpdateOne]) -> Dict[str, Any]:
        """Perform bulk updates of summaries"""
        try:
            result = self.collection.bulk_write(updates, ordered=False)
            return {
                "modified_count": result.modified_count,
                "status": "success"
            }
        except PyMongoError as e:
            self.logger.error(f"Bulk update failed: {str(e)}")
            return {
                "modified_count": 0,
                "status": "failed",
                "error": str(e)
            }

class TextPreprocessor:
    """Handles text preprocessing operations"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text"""
        if not isinstance(text, str):
            return ""
        
        text = text.strip()
        # Remove multiple newlines
        text = ' '.join(text.split())
        # Remove very short texts
        if len(text) < 100:  # Configurable threshold
            return ""
        return text

    @staticmethod
    def chunk_text(text: str, max_chunk_size: int = 1024) -> List[str]:
        """Split long text into chunks for processing"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_size = 0
        
        for word in words:
            if current_size + len(word) > max_chunk_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_size = len(word)
            else:
                current_chunk.append(word)
                current_size += len(word) + 1  # +1 for space
                
        if current_chunk:
            chunks.append(' '.join(current_chunk))
            
        return chunks

class SummarizationPipeline:
    """Main class for handling text summarization"""
    
    def __init__(self, 
                 mongo_uri: str,
                 db_name: str,
                 collection_name: str,
                 config: Optional[SummaryConfig] = None):
        """Initialize the summarization pipeline"""
        self.config = config or SummaryConfig()
        self.mongo_handler = MongoDBHandler(mongo_uri, db_name, collection_name)
        self.preprocessor = TextPreprocessor()
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize the summarizer
        try:
            self.summarizer = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                device=0 if self._is_gpu_available() else -1
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize summarizer: {str(e)}")
            raise

    @staticmethod
    def _is_gpu_available() -> bool:
        """Check if GPU is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    async def process_text(self, text: str) -> str:
        """Process and summarize text with retries"""
        for attempt in range(self.config.retry_attempts):
            try:
                cleaned_text = self.preprocessor.clean_text(text)
                if not cleaned_text:
                    return ""
                
                chunks = self.preprocessor.chunk_text(
                    cleaned_text, 
                    self.config.chunk_size
                )
                
                summaries = []
                for chunk in chunks:
                    summary = self.summarizer(
                        chunk,
                        max_length=self.config.max_length,
                        min_length=self.config.min_length,
                        do_sample=self.config.do_sample,
                        num_beams=self.config.num_beams,
                        temperature=self.config.temperature
                    )
                    summaries.append(summary[0]['summary_text'])
                
                # Combine chunk summaries
                final_summary = ' '.join(summaries)
                return final_summary
                
            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt == self.config.retry_attempts - 1:
                    raise SummarizationError(f"All retry attempts failed: {str(e)}")
                await asyncio.sleep(1)  # Wait before retry

    async def process_batch(self, articles: List[Dict]) -> List[UpdateOne]:
        """Process a batch of articles"""
        updates = []
        
        async def process_article(article: Dict) -> UpdateOne:
            try:
                summary = await self.process_text(article.get('content', ''))
                return UpdateOne(
                    {"_id": article["_id"]},
                    {
                        "$set": {
                            "summary": summary,
                            "summary_status": "success",
                            "summarized_at": datetime.utcnow(),
                            "summary_metadata": {
                                "model": "facebook/bart-large-cnn",
                                "version": "1.0",
                                "original_length": len(article.get('content', '')),
                                "summary_length": len(summary)
                            }
                        }
                    }
                )
            except Exception as e:
                self.logger.error(f"Failed to process article {article['_id']}: {str(e)}")
                return UpdateOne(
                    {"_id": article["_id"]},
                    {
                        "$set": {
                            "summary_status": "failed",
                            "summary_error": str(e),
                            "last_attempt": datetime.utcnow()
                        }
                    }
                )

        tasks = [process_article(article) for article in articles]
        updates = await asyncio.gather(*tasks)
        return updates

    async def run(self, batch_size: Optional[int] = None):
        """Main execution method"""
        batch_size = batch_size or self.config.batch_size
        start_time = time.time()
        total_processed = 0
        
        try:
            while True:
                articles = self.mongo_handler.get_unsummarized_articles(batch_size)
                print("unsummarized : ",articles)
                if not articles:
                    break

                updates = await self.process_batch(articles)
                print("updates : ",updates)
                result = self.mongo_handler.bulk_update_summaries(updates)
                
                total_processed += len(articles)
                self.logger.info(
                    f"Processed batch of {len(articles)} articles. "
                    f"Status: {result['status']}"
                )

                # Progress information
                elapsed_time = time.time() - start_time
                rate = total_processed / elapsed_time
                self.logger.info(
                    f"Total processed: {total_processed}, "
                    f"Rate: {rate:.2f} articles/second"
                )

        except KeyboardInterrupt:
            self.logger.info("Gracefully shutting down...")
        except Exception as e:
            self.logger.error(f"Pipeline error: {str(e)}")
            raise
        finally:
            self.mongo_handler.client.close()
            self.logger.info(
                f"Pipeline completed. Total processed: {total_processed} articles"
            )

def main():
    """Entry point for the summarization pipeline"""
    # Load environment variables
    mongo_uri = "mongodb://localhost:27017/"
    if not mongo_uri:
        raise ValueError("MONGODB_URI environment variable not set")

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('summarization.log')
        ]
    )

    # Initialize and run the pipeline
    config = SummaryConfig(
        max_length=600,
        min_length=50,
        batch_size=16,
        chunk_size=1024
    )
    
    pipeline = SummarizationPipeline(
        mongo_uri=mongo_uri,
        db_name='news_aggregator',
        collection_name='articles',
        config=config
    )

    asyncio.run(pipeline.run())

if __name__ == "__main__":
    main()