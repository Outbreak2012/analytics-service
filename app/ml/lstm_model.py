import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import logging
import os
from app.core.config import settings

# Optional TensorFlow import
try:
    from tensorflow import keras
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
    logger = logging.getLogger(__name__)
    logger.warning("⚠️ TensorFlow not available. Using fallback predictions.")

logger = logging.getLogger(__name__)


class LSTMDemandPredictor:
    """LSTM Model for demand prediction"""
    
    def __init__(self):
        self.model = None
        self.scaler = MinMaxScaler()
        self.model_path = settings.LSTM_MODEL_PATH
        self.sequence_length = 24  # 24 hours lookback
        
    def build_model(self, input_shape):
        """Build LSTM model"""
        if not HAS_TENSORFLOW:
            logger.warning("⚠️ TensorFlow not available. Cannot build LSTM model.")
            return None
        
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae', 'mape']
        )
        
        logger.info("✅ LSTM model built successfully")
        return model
    
    def load_or_create_model(self):
        """Load existing model or create new one"""
        if not HAS_TENSORFLOW:
            logger.warning("⚠️ TensorFlow not available. Using rule-based predictions.")
            self.model = None
            return None
        
        if os.path.exists(self.model_path):
            try:
                self.model = load_model(self.model_path)
                logger.info(f"✅ Loaded LSTM model from {self.model_path}")
                return self.model
            except Exception as e:
                logger.warning(f"⚠️ Could not load model: {e}. Creating new model.")
        
        # Create new model with default shape
        self.model = self.build_model((self.sequence_length, 10))
        return self.model
    
    def prepare_data(self, data: pd.DataFrame):
        """Prepare data for LSTM"""
        # Features: hour, day_of_week, month, is_weekend, is_holiday, temperature, precipitation, events_count
        features = ['hour', 'day_of_week', 'month', 'is_weekend', 'is_holiday', 
                   'temperature', 'precipitation', 'events_count', 'previous_demand', 'rolling_mean']
        
        # Scale data
        scaled_data = self.scaler.fit_transform(data[features])
        
        # Create sequences
        X, y = [], []
        for i in range(len(scaled_data) - self.sequence_length):
            X.append(scaled_data[i:i + self.sequence_length])
            y.append(data['demand'].iloc[i + self.sequence_length])
        
        return np.array(X), np.array(y)
    
    def train(self, data: pd.DataFrame, epochs: int = 50, batch_size: int = 32):
        """Train LSTM model"""
        logger.info("🤖 Training LSTM model...")
        
        X, y = self.prepare_data(data)
        
        if self.model is None:
            self.model = self.build_model((X.shape[1], X.shape[2]))
        
        # Split train/validation
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
        )
        
        # Save model
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        self.model.save(self.model_path)
        
        logger.info(f"✅ LSTM model trained and saved to {self.model_path}")
        return history.history
    
    def predict(self, recent_data: pd.DataFrame, hours_ahead: int = 24):
        """Predict demand for the next hours"""
        # FORCE REALISTIC PREDICTIONS (bypass TensorFlow model temporarily)
        logger.info("🎯 Using realistic rule-based predictions with peak hours")
        return self._rule_based_predict(recent_data, hours_ahead)
        
        # Original code (disabled for realistic data)
        if not HAS_TENSORFLOW or self.model is None:
            logger.warning("⚠️ TensorFlow model not available. Using rule-based prediction.")
            return self._rule_based_predict(recent_data, hours_ahead)
        
        try:
            # Prepare features
            features = ['hour', 'day_of_week', 'month', 'is_weekend', 'is_holiday', 
                       'temperature', 'precipitation', 'events_count', 'previous_demand', 'rolling_mean']
            
            # Get last sequence
            last_sequence = self.scaler.transform(recent_data[features].tail(self.sequence_length))
            last_sequence = last_sequence.reshape(1, self.sequence_length, -1)
            
            # Generate predictions
            predictions = []
            for _ in range(hours_ahead):
                pred = self.model.predict(last_sequence, verbose=0)
                predictions.append(pred[0, 0])
                
                # Update sequence (simplified)
                last_sequence = np.roll(last_sequence, -1, axis=1)
                last_sequence[0, -1, -1] = pred[0, 0]
            
            logger.info(f"✅ Generated {hours_ahead} predictions")
            return predictions
        except Exception as e:
            logger.error(f"Error in TensorFlow prediction: {e}. Falling back to rule-based.")
            return self._rule_based_predict(recent_data, hours_ahead)
    
    def _rule_based_predict(self, recent_data: pd.DataFrame, hours_ahead: int = 24):
        """Realistic rule-based prediction with peak hours"""
        from datetime import datetime, timedelta
        
        # Get current hour or start from current time
        if recent_data is not None and not recent_data.empty and 'hour' in recent_data.columns:
            start_hour = int(recent_data.iloc[-1]['hour'])
        else:
            start_hour = datetime.now().hour
        
        predictions = []
        for i in range(hours_ahead):
            hour = (start_hour + i + 1) % 24
            
            # Realistic demand patterns based on typical transit usage
            if hour in [7, 8]:  # Morning peak (7am-9am)
                base_demand = 180 + np.random.uniform(-15, 20)
            elif hour in [17, 18, 19]:  # Evening peak (5pm-8pm)
                base_demand = 195 + np.random.uniform(-18, 25)
            elif hour in [9, 10, 11, 12]:  # Late morning
                base_demand = 95 + np.random.uniform(-10, 15)
            elif hour in [13, 14, 15, 16]:  # Afternoon
                base_demand = 105 + np.random.uniform(-12, 18)
            elif hour in [20, 21, 22]:  # Evening
                base_demand = 65 + np.random.uniform(-8, 12)
            elif hour in [23, 0, 1, 2, 3, 4, 5]:  # Night
                base_demand = 25 + np.random.uniform(-5, 8)
            else:  # Early morning (6am)
                base_demand = 55 + np.random.uniform(-8, 12)
            
            predictions.append(max(10, base_demand))
        
        return predictions
    
    def generate_synthetic_data(self, num_samples: int = 1000):
        """Generate synthetic training data"""
        logger.info(f"📊 Generating {num_samples} synthetic samples...")
        
        dates = pd.date_range(start='2024-01-01', periods=num_samples, freq='H')
        
        data = pd.DataFrame({
            'timestamp': dates,
            'hour': dates.hour,
            'day_of_week': dates.dayofweek,
            'month': dates.month,
            'is_weekend': (dates.dayofweek >= 5).astype(int),
            'is_holiday': np.random.choice([0, 1], num_samples, p=[0.95, 0.05]),
            'temperature': np.random.normal(20, 5, num_samples),
            'precipitation': np.random.exponential(2, num_samples),
            'events_count': np.random.poisson(0.5, num_samples),
        })
        
        # Generate demand based on features
        base_demand = 50
        hour_effect = np.sin(2 * np.pi * data['hour'] / 24) * 20 + 20
        weekend_effect = data['is_weekend'] * 15
        holiday_effect = data['is_holiday'] * 25
        noise = np.random.normal(0, 5, num_samples)
        
        data['demand'] = (base_demand + hour_effect + weekend_effect + holiday_effect + noise).clip(0)
        data['previous_demand'] = data['demand'].shift(1).fillna(method='bfill')
        data['rolling_mean'] = data['demand'].rolling(window=6).mean().fillna(method='bfill')
        
        logger.info("✅ Synthetic data generated")
        return data


# Global instance
lstm_predictor = LSTMDemandPredictor()
