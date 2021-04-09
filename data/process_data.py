import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """Loads disaster messages and categories"""

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    df = pd.merge(messages, categories, on='id', how='left')

    return df


def clean_data(df):
    """This function takes the dataframe of disaster messages as an input,
       and performs a number of cleaning steps:

        - Splitting categories column into separate category columns
            - Converting the category strings to just numbers 0 or 1
            - Renaming column names with correct category names
            - Removing the original categories column and concatenating
              the df and categories dataframe

        - Removing duplicates
    """

    # Split categories into separate category columns
    categories = df.categories.str.split(';', expand=True)

    # Select the first row of the dataframe &
    # Extract a list of new column names for categories
    row = categories.loc[0]
    category_colnames = row.apply(lambda x: x[:-2])
    # rename the columns
    categories.columns = category_colnames

    # Converting category values to numbers 0 or 1
    for column in categories:
        categories[column] = categories[column].apply(lambda x: x[-1])
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    # drop the original categories column from `df` &
    # concatenate the original dataframe with the new `categories` dataframe
    df.drop(['categories'], axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)

    # drop duplicates
    df.drop_duplicates(keep='first', inplace=True)

    # Removing Related with values of 2 to have all values as 0 and 1
    df = df[df.related != 2]

    return df


def save_data(df, database_filepath):
    """Stores disaster messages to SQLite database"""

    engine = create_engine('sqlite:///'+database_filepath)
    df.to_sql('DisasterMessages', engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
