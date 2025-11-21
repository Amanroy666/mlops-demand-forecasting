"""Model performance monitoring"""
from typing import Dict, List
import numpy as np
from scipy import stats

class ModelMonitor:
    def __init__(self, reference_data: np.ndarray):
        self.reference_data = reference_data
        self.reference_mean = np.mean(reference_data)
        self.reference_std = np.std(reference_data)
    
    def detect_drift(self, current_data: np.ndarray) -> Dict:
        """Detect data drift using KS test"""
        ks_statistic, p_value = stats.ks_2samp(
            self.reference_data, 
            current_data
        )
        
        drift_detected = p_value < 0.05
        
        return {
            'drift_detected': drift_detected,
            'ks_statistic': float(ks_statistic),
            'p_value': float(p_value),
            'recommendation': 'Retrain model' if drift_detected else 'Continue monitoring'
        }
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate performance metrics"""
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        return {
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        }
