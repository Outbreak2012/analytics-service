import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class DBSCANUserSegmentation:
    """DBSCAN for user segmentation"""
    
    def __init__(self, eps: float = 0.5, min_samples: int = 5):
        self.eps = eps
        self.min_samples = min_samples
        self.scaler = StandardScaler()
        self.model = None
        self.labels = None
        
    def prepare_user_features(self, users_data: pd.DataFrame):
        """Prepare user features for clustering"""
        # Features: frequency, avg_spending, route_diversity, peak_hour_usage, weekend_usage
        features = [
            'usage_frequency',
            'avg_spending',
            'route_diversity',
            'peak_hour_usage_ratio',
            'weekend_usage_ratio',
            'avg_trip_duration',
            'total_transactions'
        ]
        
        # Handle missing values
        users_data = users_data.fillna(0)
        
        # Scale features
        scaled_features = self.scaler.fit_transform(users_data[features])
        
        return scaled_features, features
    
    def fit(self, users_data: pd.DataFrame):
        """Fit DBSCAN model"""
        logger.info("🤖 Fitting DBSCAN clustering model...")
        
        X, feature_names = self.prepare_user_features(users_data)
        
        # Fit DBSCAN
        self.model = DBSCAN(eps=self.eps, min_samples=self.min_samples, n_jobs=-1)
        self.labels = self.model.fit_predict(X)
        
        # Calculate metrics
        n_clusters = len(set(self.labels)) - (1 if -1 in self.labels else 0)
        n_outliers = list(self.labels).count(-1)
        
        # Calculate silhouette score (excluding outliers)
        mask = self.labels != -1
        if n_clusters > 1 and mask.sum() > 0:
            silhouette_avg = silhouette_score(X[mask], self.labels[mask])
        else:
            silhouette_avg = None
        
        logger.info(f"✅ DBSCAN completed: {n_clusters} clusters, {n_outliers} outliers")
        if silhouette_avg:
            logger.info(f"📊 Silhouette score: {silhouette_avg:.3f}")
        
        return {
            'n_clusters': n_clusters,
            'n_outliers': n_outliers,
            'silhouette_score': silhouette_avg,
            'labels': self.labels
        }
    
    def analyze_clusters(self, users_data: pd.DataFrame):
        """Analyze cluster characteristics"""
        if self.labels is None:
            raise ValueError("Model must be fitted first")
        
        users_data['cluster'] = self.labels
        clusters = []
        
        # Analyze each cluster
        unique_clusters = set(self.labels)
        if -1 in unique_clusters:
            unique_clusters.remove(-1)  # Exclude outliers from analysis
        
        for cluster_id in unique_clusters:
            cluster_data = users_data[users_data['cluster'] == cluster_id]
            
            cluster_info = {
                'cluster_id': int(cluster_id),
                'user_count': len(cluster_data),
                'avg_frequency': float(cluster_data['usage_frequency'].mean()),
                'avg_spending': float(cluster_data['avg_spending'].mean()),
                'avg_route_diversity': float(cluster_data['route_diversity'].mean()),
                'peak_hour_ratio': float(cluster_data['peak_hour_usage_ratio'].mean()),
                'weekend_ratio': float(cluster_data['weekend_usage_ratio'].mean()),
                'characteristics': self._describe_cluster(cluster_data)
            }
            
            clusters.append(cluster_info)
        
        # Outliers analysis
        outliers = users_data[users_data['cluster'] == -1]
        outliers_info = {
            'count': len(outliers),
            'avg_frequency': float(outliers['usage_frequency'].mean()) if len(outliers) > 0 else 0,
            'avg_spending': float(outliers['avg_spending'].mean()) if len(outliers) > 0 else 0
        }
        
        logger.info(f"✅ Analyzed {len(clusters)} clusters")
        return clusters, outliers_info
    
    def _describe_cluster(self, cluster_data: pd.DataFrame) -> str:
        """Generate cluster description"""
        freq = cluster_data['usage_frequency'].mean()
        spending = cluster_data['avg_spending'].mean()
        
        # Classify cluster
        if freq > cluster_data['usage_frequency'].quantile(0.75):
            freq_label = "Alta frecuencia"
        elif freq > cluster_data['usage_frequency'].quantile(0.5):
            freq_label = "Frecuencia media"
        else:
            freq_label = "Baja frecuencia"
        
        if spending > cluster_data['avg_spending'].quantile(0.75):
            spending_label = "Alto gasto"
        elif spending > cluster_data['avg_spending'].quantile(0.5):
            spending_label = "Gasto medio"
        else:
            spending_label = "Bajo gasto"
        
        return f"{freq_label} - {spending_label}"
    
    def generate_synthetic_users(self, num_users: int = 500):
        """Generate synthetic user data for testing"""
        logger.info(f"📊 Generating {num_users} synthetic users...")
        
        # Create different user profiles
        profiles = []
        
        # Profile 1: Frequent commuters (40%)
        n_commuters = int(num_users * 0.4)
        profiles.append(pd.DataFrame({
            'user_id': range(1, n_commuters + 1),
            'usage_frequency': np.random.normal(20, 3, n_commuters).clip(15, 30),
            'avg_spending': np.random.normal(150, 20, n_commuters).clip(100, 200),
            'route_diversity': np.random.normal(2, 0.5, n_commuters).clip(1, 3),
            'peak_hour_usage_ratio': np.random.normal(0.7, 0.1, n_commuters).clip(0.5, 0.9),
            'weekend_usage_ratio': np.random.normal(0.2, 0.1, n_commuters).clip(0, 0.4),
            'avg_trip_duration': np.random.normal(30, 5, n_commuters).clip(20, 45),
            'total_transactions': np.random.normal(200, 30, n_commuters).clip(150, 300)
        }))
        
        # Profile 2: Occasional users (30%)
        n_occasional = int(num_users * 0.3)
        start_id = n_commuters + 1
        profiles.append(pd.DataFrame({
            'user_id': range(start_id, start_id + n_occasional),
            'usage_frequency': np.random.normal(8, 2, n_occasional).clip(5, 12),
            'avg_spending': np.random.normal(60, 15, n_occasional).clip(30, 100),
            'route_diversity': np.random.normal(3, 1, n_occasional).clip(2, 5),
            'peak_hour_usage_ratio': np.random.normal(0.5, 0.15, n_occasional).clip(0.2, 0.7),
            'weekend_usage_ratio': np.random.normal(0.5, 0.15, n_occasional).clip(0.3, 0.7),
            'avg_trip_duration': np.random.normal(25, 8, n_occasional).clip(15, 40),
            'total_transactions': np.random.normal(80, 20, n_occasional).clip(50, 120)
        }))
        
        # Profile 3: Weekend warriors (20%)
        n_weekend = int(num_users * 0.2)
        start_id = n_commuters + n_occasional + 1
        profiles.append(pd.DataFrame({
            'user_id': range(start_id, start_id + n_weekend),
            'usage_frequency': np.random.normal(6, 1.5, n_weekend).clip(4, 9),
            'avg_spending': np.random.normal(90, 20, n_weekend).clip(50, 130),
            'route_diversity': np.random.normal(4, 1, n_weekend).clip(3, 6),
            'peak_hour_usage_ratio': np.random.normal(0.3, 0.1, n_weekend).clip(0.1, 0.5),
            'weekend_usage_ratio': np.random.normal(0.8, 0.1, n_weekend).clip(0.6, 1.0),
            'avg_trip_duration': np.random.normal(35, 10, n_weekend).clip(20, 60),
            'total_transactions': np.random.normal(60, 15, n_weekend).clip(40, 90)
        }))
        
        # Profile 4: Outliers (10%)
        n_outliers = num_users - n_commuters - n_occasional - n_weekend
        start_id = n_commuters + n_occasional + n_weekend + 1
        profiles.append(pd.DataFrame({
            'user_id': range(start_id, start_id + n_outliers),
            'usage_frequency': np.random.uniform(0, 40, n_outliers),
            'avg_spending': np.random.uniform(0, 300, n_outliers),
            'route_diversity': np.random.uniform(1, 10, n_outliers),
            'peak_hour_usage_ratio': np.random.uniform(0, 1, n_outliers),
            'weekend_usage_ratio': np.random.uniform(0, 1, n_outliers),
            'avg_trip_duration': np.random.uniform(5, 120, n_outliers),
            'total_transactions': np.random.uniform(1, 500, n_outliers)
        }))
        
        users_data = pd.concat(profiles, ignore_index=True)
        logger.info("✅ Synthetic users generated")
        return users_data


# Global instance
dbscan_segmentation = DBSCANUserSegmentation()
