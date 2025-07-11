import os 
import logging 
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import numpy as np 
import pickle 

log_path = 'logs'
os.makedirs(log_path,exist_ok=True)

logger=logging.getLogger('model_training')
logger.setLevel("DEBUG")

console_handler=logging.StreamHandler()
console_handler.setLevel("DEBUG")

log_file_path=os.path.join(log_path,'model_training.log')

file_handler=logging.FileHandler(log_file_path)
file_handler.setLevel("DEBUG")

formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(file_path:str)->pd.DataFrame:
    """load data from the csv file"""
    try:
        df=pd.read_csv(file_path)
        logger.debug("Data loaded successfully from %s", file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error("Error parsing CSV file: %s", e)
        raise
    except Exception as e:
        logger.error("An error occurred while loading data: %s", e)
        raise

def train_classifier(x_train:np.ndarray,y_train:np.ndarray,params:dict)->XGBClassifier:
    """Train a classifier using XGBoost"""
    try:
        model= XGBClassifier(n_estimators=params.get('n_estimators', 100),
                             learning_rate=params.get('learning_rate', 0.1),
                             max_depth=params.get('max_depth', 3),
                             random_state=42)
        model.fit(x_train,y_train)
        logger.debug("Model trained successfully.")
        return model
    except Exception as e:
        logger.error("An error occurred during model training: %s", e)
        raise   

def save_model(model,file_path:str)->None:
    """Save the trained model to a file"""
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)

        with open(file_path,'wb') as f:
            pickle.dump(model,f)
        logger.debug("Model saved successfully to %s", file_path)
    except Exception as e:      
        logger.error("An error occurred while saving the model: %s", e)
        raise 

def main():
    try:
        params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 3
        }
        train_data = load_data('./data/processed/train_tfidf.csv')
        X_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values

        clf = train_classifier(X_train, y_train, params)
        
        model_save_path = 'models/model.pkl'
        save_model(clf, model_save_path)

    except Exception as e:
        logger.error('Failed to complete the model building process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
