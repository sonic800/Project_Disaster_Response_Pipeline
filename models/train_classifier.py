import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import nltk
nltk.download(['punkt', 'wordnet'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import pickle


def load_data(database_filepath):
    """Loads disaster messages from SQLite database"""

    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('DisasterMessages', engine)
    X = df['message'].values
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1).values
    # Defining category names
    category_names = list(df.drop(['id', 'message', 'original', 'genre'], axis=1).columns)

    return X, Y, category_names



def tokenize(text):
    """Tokenizes text"""

    tokens = word_tokenize(text)

    # Initialize lemmatizer and lemmatize text
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens



def build_model():
    """Build ML pipeline, and set up parameters for GridSearch"""

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(KNeighborsClassifier()))
    ])

    parameters = {
    'clf__estimator__n_neighbors': [5, 10],
    'clf__estimator__weights': ['uniform', 'distance']
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1)

    return cv



def evaluate_model(model, X_test, Y_test, category_names):
    """This function reports f1 score, recall and precision for each
       category of the dataset.

       SKlearns classification_report is used by iterating through
       the category names. """

    Y_pred = model.predict(X_test)

    # Creating lists for f1-score, recall and precision
    f1_score_list = []
    recall_list = []
    precision_list = []

    # Iterating through the columns with sklearn's classification_report. Showing weighted average
    for i in range(len(category_names)):
        f1_score_list.append(str.split(classification_report(Y_test[:,i], Y_pred[:,i]))[-4:-1][2])
        recall_list.append(str.split(classification_report(Y_test[:,i], Y_pred[:,i]))[-4:-1][1])
        precision_list.append(str.split(classification_report(Y_test[:,i], Y_pred[:,i]))[-4:-1][0])

    # Print results
    for i in range(len(category_names)):
        print(category_names[i]+'\n'
        'F1-score: {}  Precision: {}  Recall: {} \n'.format(f1_score_list[i], precision_list[i], recall_list[i]))



def save_model(model, model_filepath):
    """Exports/saves the model as a pickle file"""

    Pkl_Filename = model_filepath

    with open(Pkl_Filename, 'wb') as file:
        pickle.dump(model, file)



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
