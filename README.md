![gif](https://github.com/rishabhCMS/Disaster_Response/blob/main/Peek%202020-10-25%2016-53.gif)
# Disaster Response Pipeline Project



### Aim:
The Objective of this Project was to develop a web app where the emergency workers can infer form a help message the possible categories the message belongs to.
to be able to redirect necessary support "shelter", "Food" etc


### Demo



### Dependencies
Python 3 and the following Python libraries installed:

    NumPy
    Pandas
    Matplotlib
    Json
    Plotly
    Nltk
    Flask
    Sklearn
    Sqlalchemy
    Sys
    Re
    Pickle
    
### Instructions to run:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3003/ with udacity spaceid and domian

### Results:
I was able to train the model and deploy the model


### Data Source:

[Figure Eight API](https://www.programmableweb.com/api/figure-eight-rest-api-v1)

### Acknowlegements

Udacity for the project template and Figure Eight for the data

