import logging
from typing import Dict
from app.core.config import settings
import re

# Optional transformers import
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    import torch
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

logger = logging.getLogger(__name__)


class BERTSentimentAnalyzer:
    """BERT model for sentiment analysis"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
        
    def load_model(self):
        """Load BERT model"""
        if not HAS_TRANSFORMERS:
            logger.warning("⚠️ Transformers library not available. Using rule-based sentiment.")
            self.pipeline = None
            return None
        
        try:
            logger.info("🤖 Loading BERT sentiment analysis model...")
            
            # Use sentiment-analysis pipeline with multilingual model
            self.pipeline = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                device=0 if torch.cuda.is_available() else -1
            )
            
            logger.info("✅ BERT model loaded successfully")
            return self.pipeline
        except Exception as e:
            logger.error(f"❌ Error loading BERT model: {e}")
            logger.info("⚠️ Using fallback rule-based sentiment analysis")
            self.pipeline = None
            return None
    
    def analyze(self, text: str) -> Dict:
        """Analyze sentiment of text"""
        if self.pipeline is None:
            if HAS_TRANSFORMERS:
                self.load_model()
            
            # If still None, use rule-based
            if self.pipeline is None:
                return self._rule_based_analyze(text)
        
        try:
            # Get prediction
            result = self.pipeline(text[:512])[0]  # Truncate to max length
            
            # Map labels to our categories
            sentiment_map = {
                "1 star": "NEGATIVE",
                "2 stars": "NEGATIVE",
                "3 stars": "NEUTRAL",
                "4 stars": "POSITIVE",
                "5 stars": "POSITIVE",
                "NEGATIVE": "NEGATIVE",
                "NEUTRAL": "NEUTRAL",
                "POSITIVE": "POSITIVE"
            }
            
            sentiment = sentiment_map.get(result['label'], "NEUTRAL")
            confidence = float(result['score'])
            
            # Generate detailed scores
            scores = {
                "positive": confidence if sentiment == "POSITIVE" else 0.0,
                "neutral": confidence if sentiment == "NEUTRAL" else 0.0,
                "negative": confidence if sentiment == "NEGATIVE" else 0.0
            }
            
            logger.info(f"✅ Sentiment analyzed: {sentiment} (confidence: {confidence:.2f})")
            
            return {
                "sentiment": sentiment,
                "confidence_score": confidence,
                "scores": scores
            }
            
        except Exception as e:
            logger.error(f"❌ Error analyzing sentiment: {e}. Using rule-based fallback.")
            return self._rule_based_analyze(text)
    
    def _rule_based_analyze(self, text: str) -> Dict:
        """Simple rule-based sentiment analysis"""
        text_lower = text.lower()
        
        # Positive keywords (Spanish)
        positive_words = ['excelente', 'bueno', 'genial', 'perfecto', 'recomend', 
                         'amor', 'encant', 'maravill', 'bien', 'feliz', 'gracias',
                         'puntual', 'rápido', 'eficiente', 'cómodo', 'limpio']
        
        # Negative keywords (Spanish)
        negative_words = ['mal', 'pésim', 'horrible', 'tard', 'sucio', 'grosero',
                         'nunca', 'error', 'problema', 'quej', 'demor', 'lento',
                         'incómodo', 'peor', 'desastre', 'cancel']
        
        # Count matches
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        # Determine sentiment
        if positive_count > negative_count:
            sentiment = "POSITIVE"
            confidence = min(0.6 + (positive_count * 0.1), 0.95)
            scores = {"positive": confidence, "neutral": 1 - confidence, "negative": 0.0}
        elif negative_count > positive_count:
            sentiment = "NEGATIVE"
            confidence = min(0.6 + (negative_count * 0.1), 0.95)
            scores = {"positive": 0.0, "neutral": 1 - confidence, "negative": confidence}
        else:
            sentiment = "NEUTRAL"
            confidence = 0.7
            scores = {"positive": 0.15, "neutral": 0.70, "negative": 0.15}
        
        return {
            "sentiment": sentiment,
            "confidence_score": confidence,
            "scores": scores
        }
    
    def batch_analyze(self, texts: list) -> list:
        """Analyze sentiment for multiple texts"""
        if self.pipeline is None:
            self.load_model()
        
        results = []
        for text in texts:
            result = self.analyze(text)
            results.append(result)
        
        logger.info(f"✅ Analyzed {len(texts)} texts")
        return results
    
    def get_summary_stats(self, sentiments: list) -> Dict:
        """Get summary statistics from sentiment results"""
        if not sentiments:
            return {
                "total": 0,
                "positive": 0,
                "neutral": 0,
                "negative": 0,
                "positive_ratio": 0.0,
                "negative_ratio": 0.0,
                "avg_confidence": 0.0
            }
        
        total = len(sentiments)
        positive = sum(1 for s in sentiments if s.get('sentiment') == 'POSITIVE')
        neutral = sum(1 for s in sentiments if s.get('sentiment') == 'NEUTRAL')
        negative = sum(1 for s in sentiments if s.get('sentiment') == 'NEGATIVE')
        avg_confidence = sum(s.get('confidence_score', 0) for s in sentiments) / total
        
        return {
            "total": total,
            "positive": positive,
            "neutral": neutral,
            "negative": negative,
            "positive_ratio": positive / total,
            "negative_ratio": negative / total,
            "avg_confidence": avg_confidence
        }


# Global instance
bert_analyzer = BERTSentimentAnalyzer()
