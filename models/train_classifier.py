import sys
import pandas as pd
from sqlalchemy import create_engine
import nltk
import re
import pickle

nltk.download(['punkt', 'wordnet'])

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV


def load_data(database_filepath):
    
    '''
    Load data from database
    Args:
        database_filepath: Path of sql database
    Returns:
        X: Message data (features), dataframe
        Y: Categories (target), dataframe
        category_names: Labels for 36 categories
    '''
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('df_message_table', engine)
    X = df['message']
    Y = df.iloc[:, 4:]
    category_names = list(df.columns[4:])

    return X, Y, category_names


def tokenize(text):
    
    '''
    Tokenize and clean text
    Args:
        text: original message text, string
    Returns:
        cleaned_tokens: Normalised, tokenized, and lemmatized text as,  string
    '''
    text = re.sub(r"[^a-zA-Z0-9]", ' ', text.lower())
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    cleaned_tokens = [lemmatizer.lemmatize(tok).strip() for tok in tokens]
       
    return cleaned_tokens


def build_model():
    
    '''
    Build a pipeline model optimized with gridsearch
    Args: None
    Returns: 
        cv: Results of GridSearchCV
    '''
    pipeline = Pipeline([
                        ('vect', CountVectorizer(tokenizer=tokenize)),
                        ('tfidf', TfidfTransformer()),
                        ('clf', MultiOutputClassifier(RandomForestClassifier()))
                        ])

    parameters = {'vect__ngram_range': ((1, 1), (1, 2)),
                  #'clf__estimator__n_estimators': [20, 50],
                  'clf__estimator__min_samples_split': [2, 3],
                  #'clf__estimator__criterion': ['entropy', 'gini']
                 }
    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    
    '''
    Evaluate model performance using test data
    Args: 
        model: Trained model
        X_test: Testing set (features), dataframe
        Y_test: Test true labels, dataframe
        category_names: Labels for 36 categories, string
    Returns:
        Prints accuracy and classfication report for each category
    '''
    Y_pred = model.predict(X_test)
    
    # Calculate the accuracy for each of them.
    for i in range(len(category_names)):
        print("Category:", category_names[i],"\n", classification_report(Y_test.iloc[:, i].values, Y_pred[:, i]))
        


def save_model(model, model_filepath):
    
    '''
    Save model as a pickle file 
    Args: 
        model: Trained model 
        model_filepath: Path to save the output pickle file
    Returns:
        Saved model pickle file
    '''
    pickle.dump(model, open(model_filepath, "wb"))



def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()