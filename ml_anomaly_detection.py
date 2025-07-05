import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.decomposition import PCA
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Additional ML libraries for advanced anomaly detection
try:
    from sklearn.cluster import DBSCAN
    from sklearn.svm import OneClassSVM
    from sklearn.neural_network import MLPClassifier
except ImportError:
    print("Some advanced ML libraries not available")

class PumpAnomalyDetector:
    def __init__(self, model_type='isolation_forest'):
        """
        Initialize the anomaly detection system
        
        model_type options:
        - 'isolation_forest': Good for unsupervised anomaly detection
        - 'random_forest': Supervised learning for failure classification
        - 'one_class_svm': Alternative unsupervised method
        - 'autoencoder': Neural network approach (requires tensorflow)
        """
        self.model_type = model_type
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = None
        self.model = None
        self.is_fitted = False
        
    def prepare_features(self, df):
        """Prepare and engineer features for ML model"""
        feature_df = df.copy()
        
        # Select sensor columns
        sensor_columns = [
            'discharge_pressure', 'suction_pressure', 'flow_rate', 'motor_current',
            'bearing_temp', 'motor_temp', 'vibration_x', 'vibration_y', 'vibration_z',
            'motor_rpm', 'oil_pressure', 'seal_leak_rate'
        ]
        
        # Add derived features
        if 'differential_pressure' not in feature_df.columns:
            feature_df['differential_pressure'] = (feature_df['discharge_pressure'] - 
                                                 feature_df['suction_pressure'])
        
        if 'vibration_magnitude' not in feature_df.columns:
            feature_df['vibration_magnitude'] = np.sqrt(
                feature_df['vibration_x']**2 + 
                feature_df['vibration_y']**2 + 
                feature_df['vibration_z']**2
            )
        
        if 'efficiency_indicator' not in feature_df.columns:
            feature_df['efficiency_indicator'] = (
                feature_df['flow_rate'] * feature_df['differential_pressure']
            ) / feature_df['motor_current']
        
        # Rolling statistics for trend detection (only if we have enough data)
        window_size = min(10, len(feature_df))  # Adjust window size for small datasets
        
        if len(feature_df) >= 10:
            # Normal rolling calculations with min_periods to avoid NaN
            for col in sensor_columns:
                if col in feature_df.columns:
                    feature_df[f'{col}_rolling_mean'] = feature_df[col].rolling(window=window_size, min_periods=1).mean()
                    feature_df[f'{col}_rolling_std'] = feature_df[col].rolling(window=window_size, min_periods=1).std()
                    feature_df[f'{col}_change_rate'] = feature_df[col].diff().fillna(0)
        else:
            # For small datasets, use current values as approximations
            for col in sensor_columns:
                if col in feature_df.columns:
                    feature_df[f'{col}_rolling_mean'] = feature_df[col]
                    feature_df[f'{col}_rolling_std'] = feature_df[col].std() if len(feature_df) > 1 else 0.01
                    feature_df[f'{col}_change_rate'] = 0.0  # No change for insufficient data
        
        # Time-based features
        if 'timestamp' in feature_df.columns:
            feature_df['hour'] = pd.to_datetime(feature_df['timestamp']).dt.hour
            feature_df['day_of_week'] = pd.to_datetime(feature_df['timestamp']).dt.dayofweek
            feature_df['month'] = pd.to_datetime(feature_df['timestamp']).dt.month
            
            # Cyclical encoding for time features
            feature_df['hour_sin'] = np.sin(2 * np.pi * feature_df['hour'] / 24)
            feature_df['hour_cos'] = np.cos(2 * np.pi * feature_df['hour'] / 24)
            feature_df['day_sin'] = np.sin(2 * np.pi * feature_df['day_of_week'] / 7)
            feature_df['day_cos'] = np.cos(2 * np.pi * feature_df['day_of_week'] / 7)
        
        # Handle NaN values more carefully
        # Always fill NaN values to avoid dropping rows during prediction
        feature_df = feature_df.bfill().ffill().fillna(0)
        
        # Ensure we don't return empty DataFrame
        if len(feature_df) == 0:
            raise ValueError("Feature preparation resulted in empty DataFrame. Check input data and window size.")
        
        # Select feature columns (exclude timestamp, status, etc.)
        exclude_columns = ['timestamp', 'equipment_id', 'status', 'failure_mode', 
                          'hour', 'day_of_week', 'month']
        self.feature_columns = [col for col in feature_df.columns 
                               if col not in exclude_columns]
        
        return feature_df[self.feature_columns]
    
    def create_labels(self, df):
        """Create labels for supervised learning"""
        labels = df['status'].copy()
        
        # Map status to anomaly labels
        label_mapping = {
            'Normal': 0,
            'Early_Warning': 1,
            'Degrading': 2,
            'Critical': 3,
            'Maintenance': 0  # Treat maintenance as normal
        }
        
        return labels.map(label_mapping).fillna(0)
    
    def train_model(self, df, test_size=0.2):
        """Train the anomaly detection model"""
        print(f"Training {self.model_type} model...")
        
        # Prepare features
        X = self.prepare_features(df)
        print(f"Feature matrix shape: {X.shape}")
        print(f"Features used: {len(self.feature_columns)}")
        
        if self.model_type in ['random_forest']:
            # Supervised learning
            y = self.create_labels(df.loc[X.index])
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            if self.model_type == 'random_forest':
                self.model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=15,
                    random_state=42,
                    class_weight='balanced'
                )
            
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            train_score = self.model.score(X_train_scaled, y_train)
            test_score = self.model.score(X_test_scaled, y_test)
            
            print(f"Training accuracy: {train_score:.3f}")
            print(f"Testing accuracy: {test_score:.3f}")
            
            # Cross-validation
            cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
            print(f"Cross-validation score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
            
            # Detailed classification report
            y_pred = self.model.predict(X_test_scaled)
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred))
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\nTop 10 Most Important Features:")
            print(feature_importance.head(10))
            
            return X_test, y_test, y_pred, feature_importance
            
        else:
            # Unsupervised learning
            normal_data = X[df.loc[X.index]['status'] == 'Normal']
            
            # Scale features using only normal data
            X_scaled = self.scaler.fit_transform(normal_data)
            
            if self.model_type == 'isolation_forest':
                self.model = IsolationForest(
                    contamination=0.1,  # Expected fraction of anomalies
                    random_state=42,
                    n_estimators=100
                )
            elif self.model_type == 'one_class_svm':
                self.model = OneClassSVM(nu=0.1, kernel='rbf', gamma='scale')
            
            # Train on normal data only
            self.model.fit(X_scaled)
            
            # Test on all data
            X_all_scaled = self.scaler.transform(X)
            anomaly_scores = self.model.decision_function(X_all_scaled)
            predictions = self.model.predict(X_all_scaled)
            
            # Convert predictions (-1 for anomaly, 1 for normal) to (1 for anomaly, 0 for normal)
            binary_predictions = (predictions == -1).astype(int)
            
            # Evaluate against known labels
            true_anomalies = (df.loc[X.index]['status'] != 'Normal').astype(int)
            
            print(f"Anomaly detection results:")
            print(f"True anomalies: {true_anomalies.sum()}")
            print(f"Predicted anomalies: {binary_predictions.sum()}")
            print(f"Accuracy: {(binary_predictions == true_anomalies).mean():.3f}")
            
            return X, true_anomalies, binary_predictions, anomaly_scores
        
        self.is_fitted = True
    
    def predict(self, df):
        """Make predictions on new data"""
        if self.model is None:
            raise ValueError("Model must be trained or loaded before making predictions")
        
        X = self.prepare_features(df)
        X_scaled = self.scaler.transform(X)
        
        if self.model_type == 'random_forest':
            predictions = self.model.predict(X_scaled)
            probabilities = self.model.predict_proba(X_scaled)
            return predictions, probabilities
        else:
            predictions = self.model.predict(X_scaled)
            scores = self.model.decision_function(X_scaled)
            binary_predictions = (predictions == -1).astype(int)
            return binary_predictions, scores
    
    def get_anomaly_score(self, df):
        """Get anomaly scores for ranking"""
        if self.model is None:
            raise ValueError("Model must be trained or loaded before scoring")
        
        X = self.prepare_features(df)
        X_scaled = self.scaler.transform(X)
        
        if hasattr(self.model, 'decision_function'):
            return self.model.decision_function(X_scaled)
        elif hasattr(self.model, 'predict_proba'):
            # For classification models, use max probability as confidence
            proba = self.model.predict_proba(X_scaled)
            return np.max(proba, axis=1)
        else:
            return self.model.predict(X_scaled)
    
    def save_model(self, filepath):
        """Save trained model and scaler"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'model_type': self.model_type,
            'is_fitted': self.is_fitted
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load trained model and scaler"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        self.model_type = model_data['model_type']
        self.is_fitted = model_data['is_fitted']
        print(f"Model loaded from {filepath}")

class PumpMonitoringSystem:
    def __init__(self, model_path=None):
        """Initialize the complete monitoring system"""
        self.anomaly_detector = PumpAnomalyDetector()
        if model_path and os.path.exists(model_path):
            self.anomaly_detector.load_model(model_path)
            print(f"✅ Model loaded successfully from {model_path}")
        else:
            print(f"⚠️  Model file not found: {model_path}")
        
        self.alert_thresholds = {
            'vibration_magnitude': 0.20,
            'bearing_temp': 90,
            'motor_temp': 180,
            'motor_current': 250,
            'seal_leak_rate': 5
        }
    
    def real_time_monitoring(self, current_data):
        """Process real-time sensor data"""
        # Get anomaly predictions
        predictions, scores = self.anomaly_detector.predict(current_data)
        
        # Check critical thresholds
        alerts = []
        for sensor, threshold in self.alert_thresholds.items():
            if sensor in current_data.columns:
                violations = current_data[current_data[sensor] > threshold]
                if not violations.empty:
                    alerts.append({
                        'sensor': sensor,
                        'threshold': threshold,
                        'current_value': violations[sensor].max(),
                        'severity': 'CRITICAL'
                    })
        
        # Combine ML predictions with rule-based alerts
        results = {
            'timestamp': current_data['timestamp'].iloc[-1] if 'timestamp' in current_data.columns else None,
            'ml_anomaly_detected': bool(predictions.any()),
            'anomaly_score': float(scores.max()) if len(scores) > 0 else 0,
            'rule_based_alerts': alerts,
            'overall_status': 'ANOMALY' if predictions.any() or alerts else 'NORMAL'
        }
        
        return results
    
    def generate_maintenance_report(self, monitoring_results, historical_data):
        """Generate structured data for LLM report generation"""
        report_data = {
            'equipment_id': 'PUMP_001',
            'analysis_timestamp': monitoring_results['timestamp'],
            'overall_status': monitoring_results['overall_status'],
            'anomaly_detection': {
                'ml_prediction': monitoring_results['ml_anomaly_detected'],
                'confidence_score': monitoring_results['anomaly_score'],
                'rule_violations': monitoring_results['rule_based_alerts']
            },
            'sensor_readings': {},
            'trends': {},
            'recommendations': []
        }
        
        # Add current sensor readings
        if len(historical_data) > 0:
            latest_reading = historical_data.iloc[-1]
            for column in ['discharge_pressure', 'vibration_magnitude', 'bearing_temp', 
                          'motor_current', 'seal_leak_rate']:
                if column in latest_reading:
                    report_data['sensor_readings'][column] = float(latest_reading[column])
        
        # Add trend analysis
        if len(historical_data) > 100:  # Need sufficient history
            for column in ['bearing_temp', 'vibration_magnitude', 'motor_current']:
                if column in historical_data.columns:
                    recent_trend = historical_data[column].tail(50).mean()
                    baseline = historical_data[column].head(50).mean()
                    trend_change = (recent_trend - baseline) / baseline * 100
                    report_data['trends'][column] = {
                        'baseline': float(baseline),
                        'recent': float(recent_trend),
                        'change_percent': float(trend_change)
                    }
        
        return report_data

# Example usage and testing
if __name__ == "__main__":
    # Load synthetic data (assuming it's been generated)
    try:
        df = pd.read_csv('centrifugal_pump_sensor_data.csv')
        print(f"Loaded dataset with shape: {df.shape}")
        
        # Initialize and train anomaly detector
        detector = PumpAnomalyDetector(model_type='random_forest')
        
        # Train the model
        results = detector.train_model(df, test_size=0.2)
        
        # Save the trained model
        detector.save_model('pump_anomaly_model.pkl')
        
        # Initialize monitoring system
        monitoring_system = PumpMonitoringSystem('pump_anomaly_model.pkl')
        
        # Test real-time monitoring with latest data
        latest_data = df.tail(10)
        monitoring_results = monitoring_system.real_time_monitoring(latest_data)
        
        print("\nReal-time Monitoring Results:")
        print(f"Overall Status: {monitoring_results['overall_status']}")
        print(f"ML Anomaly Detected: {monitoring_results['ml_anomaly_detected']}")
        print(f"Anomaly Score: {monitoring_results['anomaly_score']:.3f}")
        print(f"Rule-based Alerts: {len(monitoring_results['rule_based_alerts'])}")
        
        # Generate report data for LLM
        report_data = monitoring_system.generate_maintenance_report(monitoring_results, df)
        print("\nReport data generated for LLM processing")
        
    except FileNotFoundError:
        print("Please run the data generator first to create 'centrifugal_pump_sensor_data.csv'")
        print("Then run this script to train the ML model")