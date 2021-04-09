# Disaster Response Pipeline Project

### Project Overview
This project is part of Udacity Data Scientist Nanodegree. In this project a machine learning
pipeline is created to categorize emergency messages based on the needs/messages by the sender.

In this project three components are completed:
   1. ETL pipeline
   2. ML pipeline
   3. Flask Web App

At first ETL pipeline is created to extract disaster messages and categories as csv files,
cleaning the data and storing it as a dataframe to SQLite database.

ML pipeline is created to load the data from SQLite database, training Randomforest classifier,
tuning the model with GridSearchCV and exporting the final model as a pickle file.

The ML model is integrated with a Flask Web App that is used to create an interactive user-interface to classify disaster messages.

### Files
Folder "data" contains the raw data in csv files 'disaster_categories.csv' and 'disaster_messages.csv'. Additionally the folder contains script "process_data.py" that cleans the data and stores it in database, and the cleaned data in "DisasterResponse" SQLite database.

Folder "models" contains script that trains the classifier based on cleaned data in "DisasterResponse.db", and the trained model as a pickle file "classifier.pkl".

Folder "app" contains "run.py" script to run the web app, and the HTML files.

### Instructions
1. Run the following commands in the project's root directory to set up database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`

    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
