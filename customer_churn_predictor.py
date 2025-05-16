
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, precision_score, 
                            recall_score, roc_auc_score, 
                            confusion_matrix)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = "data/customer_data.csv"
MODEL_PATH = "models/churn_model.pkl"
PCA_PATH = "models/pca_model.pkl"
SCALER_PATH = "models/scaler.pkl"

class CustomerChurnPredictor:
    def __init__(self):
        self.model = None
        self.pca = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.churn_reasons = None
        
    def load_data(self):
        """Load and preprocess customer data"""
        try:
            df = pd.read_csv(DATA_PATH)
            
            if 'signup_date' in df.columns:
                df['tenure'] = (pd.to_datetime('today') - pd.to_datetime(df['signup_date'])).dt.days
            
            categoricals = ['gender', 'payment_method', 'contract_type']
            df = pd.get_dummies(df, columns=categoricals, drop_first=True)
            
            if 'churn_status' not in df.columns:
                raise ValueError("Data must contain 'churn_status' column")
            
            self.feature_names = [col for col in df.columns if col != 'churn_status']
            
            return df
        
        except Exception as e:
            print(f"Data loading error: {str(e)}")
            return None
    
    def train_model(self, df, use_pca=True):
        """Train churn prediction model"""
        try:
            X = df[self.feature_names]
            y = df['churn_status']
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, stratify=y, random_state=42
            )
        
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            smote = SMOTE(random_state=42)
            X_res, y_res = smote.fit_resample(X_train_scaled, y_train)
            
            if use_pca:
                self.pca = PCA(n_components=0.95, random_state=42)
                X_train_processed = self.pca.fit_transform(X_res)
                X_test_processed = self.pca.transform(X_test_scaled)
            else:
                X_train_processed = X_res
                X_test_processed = X_test_scaled
            
            self.model = LogisticRegression(
                penalty='l2',
                C=0.1,
                solver='liblinear',
                class_weight='balanced',
                max_iter=1000,
                random_state=42
            )
            self.model.fit(X_train_processed, y_res)
            
            y_pred = self.model.predict(X_test_processed)
            y_proba = self.model.predict_proba(X_test_processed)[:, 1]
            
            print("\nModel Performance:")
            print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
            print(f"Precision: {precision_score(y_test, y_pred):.3f}")
            print(f"Recall: {recall_score(y_test, y_pred):.3f}")
            print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.3f}")
            
            self._identify_churn_drivers(X_train_processed, y_res)
            
            return True
            
        except Exception as e:
            print(f"Training error: {str(e)}")
            return False
    
    def _identify_churn_drivers(self, X_train, y_train):
        """Analyze feature importance for churn reasons"""

        if isinstance(self.model, LogisticRegression):
            importance = np.abs(self.model.coef_[0])
            
            if self.pca:
                importance = np.dot(importance, self.pca.components_)
            
            self.churn_reasons = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            print("\nTop 5 Churn Drivers:")
            print(self.churn_reasons.head(5))
        
        elif isinstance(self.model, RandomForestClassifier):
            self.churn_reasons = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
    
    def save_models(self):
        """Save trained models and transformers"""
        try:
            Path("models").mkdir(exist_ok=True)
            joblib.dump(self.model, MODEL_PATH)
            joblib.dump(self.pca, PCA_PATH)
            joblib.dump(self.scaler, SCALER_PATH)
            print(f"\nModels saved to {MODEL_PATH}, {PCA_PATH}, {SCALER_PATH}")
        except Exception as e:
            print(f"Error saving models: {str(e)}")
    
    def visualize_churn_drivers(self):
        """Generate churn driver visualizations"""
        if self.churn_reasons is None:
            print("Train model first to identify churn drivers")
            return
            
        plt.figure(figsize=(12, 6))
        sns.barplot(
            x='importance',
            y='feature',
            data=self.churn_reasons.head(10),
            palette='coolwarm'
        )
        plt.title("Top Customer Churn Drivers")
        plt.xlabel("Impact on Churn Probability")
        plt.ylabel("Feature")
        plt.tight_layout()
        
        Path("visualization").mkdir(exist_ok=True)
        plt.savefig("visualization/top_churn_drivers.png", dpi=300)
        plt.close()
        print("\nSaved churn drivers visualization to visualization/top_churn_drivers.png")
    
    def predict_churn(self, customer_data):
        """Predict churn risk for new customers"""
        try:
            if isinstance(customer_data, dict):
                customer_data = pd.DataFrame([customer_data])
            
            X = customer_data[self.feature_names]
            X_scaled = self.scaler.transform(X)
        
            if self.pca:
                X_processed = self.pca.transform(X_scaled)
            else:
                X_processed = X_scaled
        
            probability = self.model.predict_proba(X_processed)[0, 1]
            prediction = self.model.predict(X_processed)[0]
            
            return {
                'churn_probability': float(probability),
                'churn_prediction': bool(prediction),
                'risk_level': self._get_risk_level(probability)
            }
            
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            return None
    
    def _get_risk_level(self, probability):
        """Categorize risk level"""
        if probability < 0.3:
            return "Low"
        elif probability < 0.7:
            return "Medium"
        else:
            return "High"
    
    def generate_sample_data(self, n_samples=1000):
        """Generate synthetic customer data"""
        data = {
            'customer_id': [f'C{10000+i}' for i in range(n_samples)],
            'tenure': np.random.randint(1, 365*3, n_samples),
            'monthly_charges': np.round(np.random.uniform(20, 200, n_samples), 2),
            'total_charges': np.round(np.random.uniform(50, 5000, n_samples), 2),
            'contract_type': np.random.choice(['monthly', 'yearly', 'two_year'], n_samples),
            'payment_method': np.random.choice(['credit', 'debit', 'bank_transfer'], n_samples),
            'gender': np.random.choice(['Male', 'Female'], n_samples),
            'senior_citizen': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
            'dependents': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
            'partner': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
            'internet_service': np.random.choice(['DSL', 'Fiber', 'None'], n_samples),
            'churn_status': np.random.choice([0, 1], n_samples, p=[0.85, 0.15])
        }
        return pd.DataFrame(data)

def main():
    predictor = CustomerChurnPredictor()
    
    print("Loading customer data...")
    customer_data = predictor.load_data()
    
    if customer_data is None:
        print("Generating sample data...")
        customer_data = predictor.generate_sample_data()
        customer_data.to_csv(DATA_PATH, index=False)
        print(f"Sample data saved to {DATA_PATH}")
    
    print("\nTraining churn prediction model...")
    if predictor.train_model(customer_data):
        predictor.save_models()
        
        print("\nCreating churn driver visualizations...")
        predictor.visualize_churn_drivers()
        
        test_customer = {
            'tenure': 45,
            'monthly_charges': 85.60,
            'total_charges': 3852.00,
            'contract_type': 'monthly',
            'payment_method': 'credit',
            'gender': 'Male',
            'senior_citizen': 0,
            'dependents': 0,
            'partner': 1,
            'internet_service': 'Fiber'
        }
        
        print("\nSample Prediction:")
        prediction = predictor.predict_churn(test_customer)
        print(f"Churn Probability: {prediction['churn_probability']:.1%}")
        print(f"Risk Level: {prediction['risk_level']}")
        
        print("\nProcess completed successfully!")

if __name__ == "__main__":
    main()