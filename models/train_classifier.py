# import libraries
import sys

import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])

from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier

import pickle

def load_data(database_filepath):
    ''' 
    this functions loads the data from a the sql db file

    args: 
        database_filepath: location of the db file

    Returns:
    X: the message column
    y: the categories
    category_name: the names of the categories
    '''
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table(table_name='disasterresponse_etl', con=engine)
    X = df.message.values
    remove_col = ['id', 'message', 'original', 'genre']
    y = df.loc[:, ~df.columns.isin(remove_col)]
    y.loc[:,'related'] = y['related'].replace(2,1)
    category_name = y.columns
    return X, y, category_name


def tokenize(text):
    ''' 
    
    this function applies the nlp tokenization 
    
    args:
    text: a string with untokenizated sentences
    Returns:
    clean_tokens: a list of tokenization words from input sentences
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''
    
    A machine learning pipeline takes in the message column as input and 
    output classification results on the other 36 categories in the dataset. 
    Parameters:
    None
    Returns:
    cv: a model that uses the message column to predict classifications for 36 categories
    '''
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ]))
        ])),
        
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        'features__text_pipeline__tfidf__use_idf': (True, False),
#         'clf__estimator__n_estimators': [50, 100, 200],
        'clf__estimator__min_samples_split': [2, 3, 4]
    }
    
    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    ''' 
    
    evaluates model performance based on 
    a. f1score b. precesion and recall

    args:
        model: the model to be evaluated
        X_test: test dataset (messages tokenized list)
        y_test: categories for each message
        category_names: list of categories for the messages to be classified

    prints the classification report 
    '''

    y_pred = model.predict(X_test)
    Y_pred = pd.DataFrame(data=y_pred, 
                          index=Y_test.index, 
                          columns=category_names)
    print(classification_report(Y_test, Y_pred, target_names=category_names))


def save_model(model, model_filepath):
    '''
    
    Save the model to a specified path
    
    args:
    model: ML model
    model_filepath: file path for saving the model
    Returns: none

    '''
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    '''
    
    args:
    arg1: the file path of the database
    arg2: the file path where the trained model will be saved
    Returns:
    '''
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data from db')
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test,category_names)

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
