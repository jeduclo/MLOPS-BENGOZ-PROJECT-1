import os
import pandas as pd
import joblib
from sklearn.model_selection import RandomizedSearchCV
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from config.model_params import *
from utils.common_functions import read_yaml, load_data
from scipy.stats import randint
import signal
import time

import mlflow
import mlflow.sklearn

logger = get_logger(__name__)


class TimeoutError(Exception):
    """Custom timeout exception"""
    pass


def timeout_handler(signum, frame):
    """Signal handler for timeout"""
    raise TimeoutError("Training timed out")


class ModelTraining:
    
    def __init__(self, train_path, test_path, model_output_path, max_training_time=1800):  # 30 min default
        self.train_path = train_path
        self.test_path = test_path
        self.model_output_path = model_output_path
        self.max_training_time = max_training_time  # Maximum training time in seconds
        
        # Improved parameters with early stopping and timeouts
        self.params_dist = self.get_improved_params()
        self.random_search_params = self.get_improved_search_params()
        
    def get_improved_params(self):
        """Get improved LightGBM parameters with early stopping"""
        return {
            'n_estimators': randint(50, 300),  # Reduced from potentially higher values
            'max_depth': randint(3, 8),        # Limit depth to prevent overfitting
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'num_leaves': randint(10, 100),    # Reasonable range
            'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.1, 0.5, 1.0],
            'reg_lambda': [0, 0.1, 0.5, 1.0],
            'min_child_samples': randint(10, 50),
            'early_stopping_rounds': [10, 20, 30],  # Add early stopping
            'verbose': [-1]  # Suppress verbose output
        }
    
    def get_improved_search_params(self):
        """Get improved search parameters"""
        return {
            'n_iter': 20,  # Reduced iterations for faster training
            'cv': 3,       # Reduced CV folds
            'n_jobs': -1,
            'verbose': 1,
            'random_state': 42,
            'scoring': 'f1'
        }
    
    def load_and_split_data(self):
        try:
            logger.info(f"Loading data from {self.train_path}")
            train_df = load_data(self.train_path)
            
            logger.info(f"Loading data from {self.test_path}")
            test_df = load_data(self.test_path)
            
            # Data validation
            if train_df.empty or test_df.empty:
                raise ValueError("Empty dataset detected")
            
            X_train = train_df.drop(columns=["booking_status"])
            y_train = train_df["booking_status"]
            
            X_test = test_df.drop(columns=["booking_status"])
            y_test = test_df["booking_status"]
            
            # Check for class imbalance
            class_distribution = y_train.value_counts()
            logger.info(f"Class distribution: {class_distribution.to_dict()}")
            
            logger.info("Data split successfully for Model Training")
            
            return X_train, y_train, X_test, y_test
                        
        except Exception as e:
            logger.error(f"Error during loading and splitting step {e}")
            raise CustomException("Error while loading and splitting data", e)
    
    def train_lgbm_with_timeout(self, X_train, y_train):
        """Train LightGBM with timeout and early stopping"""
        try:
            # Set up timeout signal
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(self.max_training_time)
            
            logger.info("Initializing our model with improved parameters")
            
            # Base model with early stopping
            lgbm_model = lgb.LGBMClassifier(
                random_state=self.random_search_params["random_state"],
                n_jobs=-1,
                verbose=-1,  # Suppress warnings
                force_col_wise=True,  # Optimize for performance
                objective='binary',
                metric='binary_logloss'
            )
            
            logger.info("Starting hyperparameter tuning with timeout protection")
            
            # Create validation set for early stopping
            from sklearn.model_selection import train_test_split
            X_train_split, X_val, y_train_split, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
            )
            
            # Modified RandomizedSearchCV with fit_params for early stopping
            random_search = RandomizedSearchCV(
                estimator=lgbm_model,
                param_distributions=self.params_dist,
                n_iter=self.random_search_params["n_iter"],
                cv=self.random_search_params["cv"],
                n_jobs=1,  # Use 1 job to better control timeout
                verbose=self.random_search_params["verbose"],
                random_state=self.random_search_params["random_state"],
                scoring=self.random_search_params["scoring"]
            )
            
            start_time = time.time()
            logger.info("Starting hyperparameter tuning!")
            
            # Fit with early stopping callback
            random_search.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric='binary_logloss',
                callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)]
            )
            
            training_time = time.time() - start_time
            logger.info(f"Hyperparameter tuning completed in {training_time:.2f} seconds!")
            
            # Cancel timeout
            signal.alarm(0)
            
            best_params = random_search.best_params_
            best_lgbm_model = random_search.best_estimator_
            
            logger.info(f"Best parameters: {best_params}")
            
            return best_lgbm_model
            
        except TimeoutError:
            logger.error(f"Training timed out after {self.max_training_time} seconds")
            signal.alarm(0)
            # Return a simple model as fallback
            return self.train_simple_fallback_model(X_train, y_train)
            
        except Exception as e:
            signal.alarm(0)  # Cancel timeout
            logger.error(f"Error while training model {e}")
            # Try fallback model
            logger.info("Attempting fallback training...")
            return self.train_simple_fallback_model(X_train, y_train)
    
    def train_simple_fallback_model(self, X_train, y_train):
        """Train a simple model as fallback"""
        try:
            logger.info("Training simple fallback model")
            fallback_model = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
                verbose=-1,
                force_col_wise=True
            )
            fallback_model.fit(X_train, y_train)
            logger.info("Fallback model trained successfully")
            return fallback_model
            
        except Exception as e:
            logger.error(f"Even fallback model failed: {e}")
            raise CustomException("All training methods failed", e)
    
    def evaluate_model(self, model, X_test, y_test):
        try:
            logger.info("Evaluating our model")
            y_pred = model.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            logger.info(f"Accuracy Score: {accuracy:.4f}")
            logger.info(f"Precision Score: {precision:.4f}")
            logger.info(f"Recall Score: {recall:.4f}")
            logger.info(f"F1 Score: {f1:.4f}")
            
            return {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1
            }
            
        except Exception as e:
            logger.error(f"Error while evaluating model {e}")
            raise CustomException("Failed to evaluate model", e)
        
    def save_model(self, model):
        try:
            logger.info("Making directory to save model")
            
            os.makedirs(os.path.dirname(self.model_output_path), exist_ok=True)
            
            logger.info("Saving the model")
            
            joblib.dump(model, self.model_output_path)
            logger.info(f"Model saved to {self.model_output_path}")
                    
        except Exception as e:
            logger.error(f"Error while saving model {e}")
            raise CustomException("Failed to save model", e)
        
    def run(self):
        try:
            # Set MLflow tracking URI and experiment
            mlflow.set_tracking_uri("./mlruns")  # Local tracking
            mlflow.set_experiment("booking_prediction")
            
            with mlflow.start_run():
                logger.info("Starting model training pipeline with timeout protection")
                
                logger.info("Starting MLflow experimentation")
                
                # Log dataset info
                logger.info("Logging training and testing dataset info to MLflow")
                mlflow.log_param("train_path", self.train_path)
                mlflow.log_param("test_path", self.test_path)
                mlflow.log_param("max_training_time", self.max_training_time)
                
                # Load and validate data
                X_train, y_train, X_test, y_test = self.load_and_split_data()
                
                # Log data statistics
                mlflow.log_param("train_samples", len(X_train))
                mlflow.log_param("test_samples", len(X_test))
                mlflow.log_param("n_features", X_train.shape[1])
                
                # Train model with timeout protection
                best_lgbm_model = self.train_lgbm_with_timeout(X_train, y_train)
                
                # Evaluate model
                metrics = self.evaluate_model(best_lgbm_model, X_test, y_test)
                
                # Save model
                self.save_model(best_lgbm_model)
                
                # Log to MLflow
                logger.info("Logging model to MLflow")
                mlflow.sklearn.log_model(best_lgbm_model, "model")
                
                logger.info("Logging parameters and metrics to MLflow")
                mlflow.log_params(best_lgbm_model.get_params())
                mlflow.log_metrics(metrics)
                
                # Log model file
                mlflow.log_artifact(self.model_output_path, artifact_path="saved_models")
                
                logger.info("Model training successfully completed!")
                
                return best_lgbm_model, metrics
            
        except Exception as e:
            logger.error(f"Error in model training pipeline: {e}")
            raise CustomException("Model training pipeline failed!", e)


if __name__ == "__main__":
    # Initialize with 30-minute timeout
    trainer = ModelTraining(
        PROCESSED_TRAIN_DATA_PATH, 
        PROCESSED_TEST_DATA_PATH, 
        MODEL_OUTPUT_PATH,
        max_training_time=1800  # 30 minutes
    )
    
    try:
        model, metrics = trainer.run()
        print("Training completed successfully!")
        print(f"Final metrics: {metrics}")
    except Exception as e:
        print(f"Training failed: {e}")
        exit(1)