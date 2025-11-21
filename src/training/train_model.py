"""Model training pipeline with MLflow"""
import mlflow
import mlflow.sklearn
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
import pandas as pd

class DemandForecastTrainer:
    def __init__(self, config):
        self.config = config
        mlflow.set_tracking_uri(config['mlflow_uri'])
    
    def train(self, df: pd.DataFrame):
        """Train demand forecasting model"""
        with mlflow.start_run():
            # Feature engineering
            X, y = self.prepare_features(df)
            
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=5)
            
            # Train XGBoost model
            model = XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6
            )
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                model.fit(X_train, y_train)
                val_score = model.score(X_val, y_val)
                mlflow.log_metric("val_r2", val_score)
            
            # Log model
            mlflow.sklearn.log_model(model, "model")
            mlflow.log_params(model.get_params())
            
            return model
    
    def prepare_features(self, df):
        """Create time series features"""
        df['lag_1'] = df['demand'].shift(1)
        df['lag_7'] = df['demand'].shift(7)
        df['rolling_mean_7'] = df['demand'].rolling(7).mean()
        
        features = ['lag_1', 'lag_7', 'rolling_mean_7']
        X = df[features].dropna()
        y = df.loc[X.index, 'demand']
        
        return X, y
