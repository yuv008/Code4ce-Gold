import os
import nltk
import logging
from typing import Dict, List, Optional, Union
from datetime import datetime
import pandas as pd
from transformers import pipeline
from nltk.tokenize import sent_tokenize
import numpy as np
from pathlib import Path

class SentimentAnalyzer:
    def __init__(self):
        """Initialize the sentiment analyzer with required NLTK resources and models"""
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize NLTK resources
        self._initialize_nltk()
        
        # Initialize the transformer pipeline
        try:
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
            )
        except Exception as e:
            self.logger.error(f"Error initializing sentiment pipeline: {str(e)}")
            raise

    def _initialize_nltk(self):
        """Initialize NLTK resources with proper error handling"""
        try:
            # Create NLTK data directory if it doesn't exist
            nltk_data_dir = Path.home() / 'nltk_data'
            nltk_data_dir.mkdir(parents=True, exist_ok=True)
            
            # Download required NLTK data
            nltk.download('punkt', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            
            # Verify the downloads
            if not nltk.data.find('tokenizers/punkt'):
                raise RuntimeError("Failed to download punkt tokenizer")
                
        except Exception as e:
            self.logger.error(f"Error initializing NLTK resources: {str(e)}")
            raise

    def analyze_text(self, text: str) -> Dict:
        """
        Analyze sentiment of a single text
        
        Args:
            text (str): Input text for sentiment analysis
            
        Returns:
            Dict: Sentiment analysis results
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
            # Split text into sentences
            sentences = sent_tokenize(text)
            
            # Analyze each sentence
            sentence_results = []
            overall_score = 0
            
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
            
            # Calculate overall metrics
            num_sentences = len(sentence_results)
            if num_sentences > 0:
                average_score = overall_score / num_sentences
                magnitude = sum(abs(s['score']) for s in sentence_results) / num_sentences
                
                # Determine overall label
                if abs(average_score) < 0.1:
                    label = 'neutral'
                else:
                    label = 'positive' if average_score > 0 else 'negative'
                
                confidence = np.mean([s['confidence'] for s in sentence_results])
            else:
                average_score = 0
                magnitude = 0
                label = 'neutral'
                confidence = 0
            
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

    def analyze_trend(self, texts: List[str], timestamps: List[datetime]) -> Dict:
        """
        Analyze sentiment trends over time
        
        Args:
            texts (List[str]): List of texts to analyze
            timestamps (List[datetime]): Corresponding timestamps
            
        Returns:
            Dict: Trend analysis results
        """
        if not texts or not timestamps or len(texts) != len(timestamps):
            return self._empty_trend_result()
            
        try:
            # Create DataFrame with timestamps and sentiment scores
            results = [self.analyze_text(text) for text in texts]
            df = pd.DataFrame({
                'timestamp': timestamps,
                'score': [r['score'] for r in results],
                'label': [r['label'] for r in results]
            })
            
            df = df.sort_values('timestamp')
            
            # Calculate trend metrics
            rolling_avg = df['score'].rolling(window=min(3, len(df)), min_periods=1).mean()
            
            # Calculate trend direction and strength
            if len(df) > 1:
                slope = np.polyfit(range(len(df)), df['score'], 1)[0]
                trend_direction = 'up' if slope > 0 else 'down' if slope < 0 else 'stable'
                trend_strength = abs(slope)
                stability = 1 - df['score'].std() if len(df) > 1 else 1
            else:
                trend_direction = 'stable'
                trend_strength = 0
                stability = 1
            
            # Calculate sentiment distribution
            sentiment_counts = df['label'].value_counts(normalize=True)
            
            return {
                'overall_trend': {
                    'direction': trend_direction,
                    'strength': trend_strength,
                    'stability': stability
                },
                'rolling_averages': rolling_avg.tolist(),
                'timestamps': df['timestamp'].tolist(),
                'sentiment_distribution': {
                    'positive': sentiment_counts.get('positive', 0),
                    'neutral': sentiment_counts.get('neutral', 0),
                    'negative': sentiment_counts.get('negative', 0)
                },
                'volatility': df['score'].std() if len(df) > 1 else 0,
                'extreme_points': {
                    'most_positive': {
                        'timestamp': df.loc[df['score'].idxmax(), 'timestamp'],
                        'score': df['score'].max()
                    },
                    'most_negative': {
                        'timestamp': df.loc[df['score'].idxmin(), 'timestamp'],
                        'score': df['score'].min()
                    }
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in trend analysis: {str(e)}")
            return self._empty_trend_result()

    def _empty_trend_result(self) -> Dict:
        """Return empty trend result structure"""
        return {
            'overall_trend': {
                'direction': 'stable',
                'strength': 0,
                'stability': 1
            },
            'rolling_averages': [],
            'timestamps': [],
            'sentiment_distribution': {
                'positive': 0,
                'neutral': 0,
                'negative': 0
            },
            'volatility': 0,
            'extreme_points': {
                'most_positive': None,
                'most_negative': None
            }
        }

def main():
    """Main execution function"""
    try:
        analyzer = SentimentAnalyzer()
        
        # Example text for testing
        test_text = "I love this product! It works great. Though the price is a bit high."
        
        # Single text analysis
        single_result = analyzer.analyze_text(test_text)
        print("Single text analysis:", single_result)
        
        # Trend analysis example
        test_texts = [
            "The product is amazing!",
            "It's working okay, but has some issues.",
            "Customer service was helpful today.",
            "Very disappointed with the latest update."
        ]
        test_timestamps = [
            datetime(2024, 10, 19, 10),
            datetime(2024, 10, 19, 11),
            datetime(2024, 10, 19, 12),
            datetime(2024, 10, 19, 13)
        ]
        
        trend_result = analyzer.analyze_trend(test_texts, test_timestamps)
        print("Trend analysis:", trend_result)
        
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()