import os 
import logging  
import pandas as pd 
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split
from nltk.stem.porter import PorterStemmer 
from nltk.corpus import stopwords 
import string 
import nltk 

nltk.download('stopwords')
nltk.download('punkt')

log_dir='logs'
os.makedirs(log_dir, exist_ok=True)

logger=logging.getLogger('preprocessing')
logger.setLevel('DEBUG')

console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path=os.path.join(log_dir,'preprocessing.log')
file_handler=logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter) 
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def preprocess_text(text:str):
    ps=PorterStemmer()

    text=text.lower()

    text=nltk.word_tokenize(text)

    text=[word for word in text if word.isalnum()]

    text=[word for word in text if word not in stopwords.words('english') and word not in string.punctuation]

    text=[ps.stem(word) for word in text]
    return ' '.join(text)


def preprocess_data(df,text_column='text',target_column='label'):
    try:
        logger.debug("Starting data preprocessing...")

        encoder= LabelEncoder()
        df[target_column]=encoder.fit_transform(df[target_column])
        logger.debug("Label encoding completed.")

        df=df.drop_duplicates(keep="first")
        logger.debug(f"Removed duplicates, remaining rows: {len(df)}")  
        df.loc[:,text_column]=df[text_column].apply(preprocess_text)
        logger.debug("Text preprocessing completed.")
        return df 
    except Exception as e:
        logger.error(f"An error occurred during preprocessing: {e}")
        raise


def main():
    try:
        # Check if train.csv and test.csv exist, if not use the main dataset
        train_path = 'data/raw/train.csv'
        test_path = 'data/raw/test.csv'
        main_dataset_path = 'data/raw/spam_ham_dataset.csv'
        
        if os.path.exists(train_path) and os.path.exists(test_path):
            train_data=pd.read_csv(train_path)
            test_data=pd.read_csv(test_path)
        elif os.path.exists(main_dataset_path):
            # Split the main dataset into train and test
            data = pd.read_csv(main_dataset_path)
            from sklearn.model_selection import train_test_split
            train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
            logger.debug("Split main dataset into train and test sets.")
        else:
            raise FileNotFoundError("No suitable data files found.")
            
        logger.debug("Data loaded successfully.")

        train_preprocessed_data=preprocess_data(train_data)
        test_preprocessed_data=preprocess_data(test_data)
        logger.debug("Data preprocessing completed successfully.")

        data_path=os.path.join('./data','interim')
        os.makedirs(data_path, exist_ok=True)

        train_preprocessed_data.to_csv(os.path.join(data_path,'train_processed.csv'), index=False)
        test_preprocessed_data.to_csv(os.path.join(data_path,'test_processed.csv'), index=False)
        logger.debug("Preprocessed data saved successfully.")

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
    except pd.errors.EmptyDataError as e:
        logger.error(f"Empty data error: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        raise

if __name__=="__main__":
    main()


      