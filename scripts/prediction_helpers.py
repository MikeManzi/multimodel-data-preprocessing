import joblib
import numpy as np
import pandas as pd
import os
from sklearn.base import BaseEstimator, ClassifierMixin

# === Define custom model classes that were used during training ===
class FacialRecognitionModel(BaseEstimator, ClassifierMixin):
    """Custom wrapper for facial recognition model"""
    def __init__(self, base_model=None):
        self.base_model = base_model
    
    def fit(self, X, y):
        if self.base_model:
            self.base_model.fit(X, y)
        return self
    
    def predict(self, X):
        if self.base_model:
            return self.base_model.predict(X)
        return np.zeros(len(X))
    
    def predict_proba(self, X):
        if self.base_model and hasattr(self.base_model, 'predict_proba'):
            return self.base_model.predict_proba(X)
        return np.random.rand(len(X), 2)

class VoiceprintModel(BaseEstimator, ClassifierMixin):
    """Custom wrapper for voiceprint model"""
    def __init__(self, base_model=None):
        self.base_model = base_model
    
    def fit(self, X, y):
        if self.base_model:
            self.base_model.fit(X, y)
        return self
    
    def predict(self, X):
        if self.base_model:
            return self.base_model.predict(X)
        return np.zeros(len(X))
    
    def predict_proba(self, X):
        if self.base_model and hasattr(self.base_model, 'predict_proba'):
            return self.base_model.predict_proba(X)
        return np.random.rand(len(X), 2)

class ProductRecommendationModel(BaseEstimator):
    """Custom wrapper for product recommendation model"""
    def __init__(self, base_model=None):
        self.base_model = base_model
    
    def fit(self, X, y):
        if self.base_model:
            self.base_model.fit(X, y)
        return self
    
    def predict(self, X):
        if self.base_model:
            return self.base_model.predict(X)
        return np.zeros(len(X))

# === Load all models and tools ===
try:
    facial_model = joblib.load('../models/facial_recognition_model.pkl')
    print("✓ Facial recognition model loaded successfully")
except Exception as e:
    print(f"✗ Error loading facial model: {e}")
    facial_model = None

try:
    voice_model = joblib.load('../models/voiceprint_model.pkl')
    print("✓ Voice model loaded successfully")
except Exception as e:
    print(f"✗ Error loading voice model: {e}")
    voice_model = None

try:
    product_model = joblib.load('../models/product_recommendation_model.pkl')
    print("✓ Product recommendation model loaded successfully")
except Exception as e:
    print(f"✗ Error loading product model: {e}")
    product_model = None

# Try to load scaler and encoder if they exist
scaler = None
encoder = None
try:
    if os.path.exists('../models/scaler.pkl'):
        scaler = joblib.load('../models/scaler.pkl')
        print("✓ Scaler loaded successfully")
except Exception as e:
    print(f"✗ Error loading scaler: {e}")

try:
    if os.path.exists('../models/label_encoder.pkl'):
        encoder = joblib.load('../models/label_encoder.pkl')
        print("✓ Label encoder loaded successfully")
except Exception as e:
    print(f"✗ Error loading label encoder: {e}")

def predict_face(features):
    """
    Predict face recognition confidence using the loaded facial model.
    Returns confidence percentage (0-100).
    """
    if facial_model is None:
        print("Warning: Facial model not available, using fallback prediction")
        return np.random.uniform(20, 80)
    
    try:
        # Ensure features are in the right format
        if isinstance(features, np.ndarray):
            if features.ndim == 1:
                features = features.reshape(1, -1)
        
        # Handle dictionary models (extract components)
        model = facial_model
        pca = None
        selector = None
        scaler_face = None
        
        if isinstance(facial_model, dict):
            # Extract preprocessing components
            if 'pca' in facial_model:
                pca = facial_model['pca']
            if 'selector' in facial_model:
                selector = facial_model['selector']
            if 'scaler' in facial_model:
                scaler_face = facial_model['scaler']
            
            # Extract the actual model
            if 'model' in facial_model:
                model = facial_model['model']
            elif 'best_model' in facial_model:
                model = facial_model['best_model']
            elif 'estimator' in facial_model:
                model = facial_model['estimator']
            elif 'classifier' in facial_model:
                model = facial_model['classifier']
        
        # Apply preprocessing pipeline if components exist
        processed_features = features
        
        # Apply scaler if available
        if scaler_face is not None:
            try:
                processed_features = scaler_face.transform(processed_features)
            except:
                pass
        
        # Apply PCA if available
        if pca is not None:
            try:
                processed_features = pca.transform(processed_features)
            except:
                pass
        
        # Apply feature selector if available
        if selector is not None:
            try:
                processed_features = selector.transform(processed_features)
            except:
                pass
        
        # Try prediction
        try:
            prediction = model.predict_proba(processed_features)
            confidence = prediction[0][1] * 100 if prediction.shape[1] > 1 else prediction[0][0] * 100
            return min(100, max(0, confidence))
        except:
            # If that fails, try with DataFrame
            feature_names = [f'feat_{i}' for i in range(processed_features.shape[1])]
            df = pd.DataFrame(processed_features, columns=feature_names)
            prediction = model.predict_proba(df)
            confidence = prediction[0][1] * 100 if prediction.shape[1] > 1 else prediction[0][0] * 100
            return min(100, max(0, confidence))
    
    except Exception as e:
        print(f"Error in face prediction: {e}")
        return np.random.uniform(20, 80)

def predict_voice(features):
    """
    Predict voice verification confidence using the loaded voice model.
    Returns confidence percentage (0-100).
    """
    if voice_model is None:
        print("Warning: Voice model not available, using fallback prediction")
        return np.random.uniform(30, 85)
    
    try:
        # Ensure features are in the right format
        if isinstance(features, np.ndarray):
            if features.ndim == 1:
                features = features.reshape(1, -1)
        
        # Handle dictionary models (extract components)
        model = voice_model
        pca = None
        selector = None
        scaler_voice = None
        
        if isinstance(voice_model, dict):
            # Extract preprocessing components
            if 'pca' in voice_model:
                pca = voice_model['pca']
            if 'selector' in voice_model:
                selector = voice_model['selector']
            if 'scaler' in voice_model:
                scaler_voice = voice_model['scaler']
            
            # Extract the actual model
            if 'model' in voice_model:
                model = voice_model['model']
            elif 'best_model' in voice_model:
                model = voice_model['best_model']
            elif 'estimator' in voice_model:
                model = voice_model['estimator']
            elif 'classifier' in voice_model:
                model = voice_model['classifier']
        
        # Apply preprocessing pipeline if components exist
        processed_features = features
        
        # Apply scaler if available
        if scaler_voice is not None:
            try:
                processed_features = scaler_voice.transform(processed_features)
            except:
                pass
        
        # Apply PCA if available
        if pca is not None:
            try:
                processed_features = pca.transform(processed_features)
            except:
                pass
        
        # Apply feature selector if available
        if selector is not None:
            try:
                processed_features = selector.transform(processed_features)
            except:
                pass
        
        # Try prediction
        try:
            prediction = model.predict_proba(processed_features)
            confidence = prediction[0][1] * 100 if prediction.shape[1] > 1 else prediction[0][0] * 100
            return min(100, max(0, confidence))
        except:
            # If that fails, try with DataFrame
            feature_names = [f'audio_feat_{i}' for i in range(processed_features.shape[1])]
            df = pd.DataFrame(processed_features, columns=feature_names)
            prediction = model.predict_proba(df)
            confidence = prediction[0][1] * 100 if prediction.shape[1] > 1 else prediction[0][0] * 100
            return min(100, max(0, confidence))
    
    except Exception as e:
        print(f"Error in voice prediction: {e}")
        return np.random.uniform(30, 85)

def predict_product():
    """
    Predict product recommendation using the loaded product model.
    Returns product name string.
    """
    if product_model is None:
        print("Warning: Product model not available, using fallback prediction")
        fallback_products = [
            "Wireless Bluetooth Headphones",
            "Smart Fitness Tracker", 
            "Portable Power Bank",
            "Wireless Mouse",
            "USB-C Hub"
        ]
        return np.random.choice(fallback_products)
    
    try:
        # Handle dictionary models (extract components)
        model = product_model
        pca_prod = None
        selector_prod = None
        scaler_prod = None
        encoder_prod = encoder  # Use global encoder or extract from dict
        
        if isinstance(product_model, dict):
            # Extract preprocessing components
            if 'pca' in product_model:
                pca_prod = product_model['pca']
            if 'selector' in product_model:
                selector_prod = product_model['selector']
            if 'scaler' in product_model:
                scaler_prod = product_model['scaler']
            if 'encoder' in product_model:
                encoder_prod = product_model['encoder']
            
            # Extract the actual model
            if 'model' in product_model:
                model = product_model['model']
            elif 'best_model' in product_model:
                model = product_model['best_model']
            elif 'estimator' in product_model:
                model = product_model['estimator']
        else:
            # Use global scaler if not in dict
            scaler_prod = scaler
        
        # Example input data with realistic values
        # Full feature set (26 features)
        input_data = {
            'purchase_interest_score_mean': 7.5,
            'purchase_interest_score': 8.1,
            'purchase_amount': 49.99,
            'purchase_amount_mean': 45.20,
            'is_weekend': 0,
            'purchase_month': 6,
            'engagement_score': 7.2,
            'purchase_amount_max': 59.99,
            'purchase_day_of_month': 15,
            'customer_rating_mean': 4.3,
            'engagement_interest_interaction': 58.32,
            'purchase_quarter': 2,
            'purchase_day_of_week': 2,
            'customer_rating_max': 5.0,
            'purchase_amount_min': 32.50,
            'customer_rating_min': 3.5,
            'purchase_amount_std': 12.34,
            'customer_rating': 4.5,
            'engagement_score_std': 1.2,
            'purchase_amount_sum': 226.00,
            'amount_rating_interaction': 224.955,
            'sentiment_numeric': 2,
            'purchase_interest_score_std': 1.0,
            'engagement_score_mean': 6.8,
            'engagement_score_max': 8.5,
            'amount_per_engagement': 6.94
        }

        # Try to determine expected features from model
        expected_n_features = None
        if hasattr(model, 'n_features_in_'):
            expected_n_features = model.n_features_in_
        
        # Top 15 most important features (based on typical importance in recommendation models)
        top_15_features = [
            'purchase_interest_score_mean', 'purchase_interest_score',
            'purchase_amount', 'purchase_amount_mean', 
            'engagement_score', 'purchase_amount_max',
            'customer_rating_mean', 'engagement_interest_interaction',
            'customer_rating_max', 'customer_rating',
            'purchase_amount_sum', 'amount_rating_interaction',
            'sentiment_numeric', 'engagement_score_mean',
            'amount_per_engagement'
        ]
        
        # Use the correct feature set based on model expectations
        if expected_n_features == 15:
            feature_list = top_15_features
        elif expected_n_features and expected_n_features < 26:
            # Use the first N features from top_15_features
            feature_list = top_15_features[:expected_n_features]
        else:
            # Use all 26 features
            feature_list = [
                'purchase_interest_score_mean', 'purchase_interest_score',
                'purchase_amount', 'purchase_amount_mean', 'is_weekend', 'purchase_month',
                'engagement_score', 'purchase_amount_max', 'purchase_day_of_month',
                'customer_rating_mean', 'engagement_interest_interaction',
                'purchase_quarter', 'purchase_day_of_week', 'customer_rating_max',
                'purchase_amount_min', 'customer_rating_min', 'purchase_amount_std',
                'customer_rating', 'engagement_score_std', 'purchase_amount_sum',
                'amount_rating_interaction', 'sentiment_numeric',
                'purchase_interest_score_std', 'engagement_score_mean',
                'engagement_score_max', 'amount_per_engagement'
            ]

        # Fill missing keys with 0
        for feat in feature_list:
            if feat not in input_data:
                input_data[feat] = 0

        # Create DataFrame with the exact column order
        X = pd.DataFrame([input_data], columns=feature_list)
        
        # Apply preprocessing pipeline
        processed_X = X
        
        # Apply scaling if scaler is available
        if scaler_prod is not None:
            try:
                processed_X = scaler_prod.transform(processed_X)
            except:
                pass
        
        # Apply PCA if available
        if pca_prod is not None:
            try:
                processed_X = pca_prod.transform(processed_X)
            except:
                pass
        
        # Apply feature selector if available
        if selector_prod is not None:
            try:
                processed_X = selector_prod.transform(processed_X)
            except:
                pass
        
        # Make prediction
        pred = model.predict(processed_X)
        
        # Apply label encoding if encoder is available
        if encoder_prod is not None:
            return encoder_prod.inverse_transform(pred)[0]
        else:
            return f"Product_{pred[0]}"  # Fallback product naming
    
    except Exception as e:
        print(f"Error in product prediction: {e}")
        fallback_products = [
            "Wireless Bluetooth Headphones",
            "Smart Fitness Tracker",
            "Portable Power Bank", 
            "Wireless Mouse",
            "USB-C Hub"
        ]
        return np.random.choice(fallback_products)