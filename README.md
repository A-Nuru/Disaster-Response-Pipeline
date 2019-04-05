# Disaster Response Pipeline Project
## Table of Contents
1. [Installation](https://github.com/A-Nuru/Disaster-Response-Pipeline#Installation)
2. [Project Motivation](https://github.com/A-Nuru/Disaster-Response-Pipeline#Project-Motivation)
3. [File Descriptions](https://github.com/A-Nuru/Disaster-Response-Pipeline#File-Descriptions)
4. [Instructions](https://github.com/A-Nuru/Disaster-Response-Pipeline#Instructions)
5. [Results](https://github.com/A-Nuru/Disaster-Response-Pipeline#Results)
6. [Licensing, Authors, Acknowledgements](https://github.com/A-Nuru/Disaster-Response-Pipeline#Licensing-Authors-Acknowledgements)

## Installation
This project requires **Python 3.x** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org)
- [matplotlib](http://matplotlib.org/)
- [scikit-learn](http://scikit-learn.org/stable/)

It is recommended to install [Anaconda](https://www.continuum.io/downloads), a pre-packaged Python distribution that contains all of the necessary libraries and software for this project. 
You might also need to have software installed to run and execute an [iPython Notebook](http://ipython.org/notebook.html)

## Project Motivation
In this project, we analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages.The data set contains real messages that were sent during disaster events. We will build and train a machine learning pipeline to categorize these events so that the message can be sent to an appropriate disaster relief agency for prompt action.

The project includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data.


## File Descriptions
The files are organised into folders as shown below:
* app folder containing
    - template consisting of master.html - main page of web app and go.html - classification result page of web app
    - run.py  # Flask file that runs app

* data folder comprised of
    - disaster_categories.csv  - data to process 
    - disaster_messages.csv  - data to process
    - process_data.py
    - InsertDatabaseName.db   - database to save clean data to

* models folder which contains
    - train_classifier.py
    - classifier.pkl  - saved model 
* A README.md file

## Instructions
1. In the command line, run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`
3. To get WORKSPACEID and WORKSPACEDOMAIN, open another Terminal Window and run
    `env|grep WORK` 
4. Go to https://SPACEID-3001.SPACEDOMAIN to render the website

## Results
An application and a web app including data visualizations where an emergency worker can input a new disaster message and get classified result in several categories. Checkout the main files [here.](https://github.com/A-Nuru/Disaster-Response-Pipeline)

## Licensing, Authors, Acknowledgements
The license of this project can be found [here.](https://github.com/A-Nuru/Disaster-Response-Pipeline/blob/master/LICENSE)
Credits must be given to Figure8 for providing the data to Udacity.



![DRP1](https://user-images.githubusercontent.com/45924101/55646154-c0c2f500-57da-11e9-96b3-9e5aea1bbc3a.PNG)


![DRP2](https://user-images.githubusercontent.com/45924101/55646252-0089dc80-57db-11e9-8f49-58caa3e0729c.PNG)


![DRP3](https://user-images.githubusercontent.com/45924101/55646273-097aae00-57db-11e9-8c53-59114fb5d958.PNG)


![DRP4](https://user-images.githubusercontent.com/45924101/55646284-0e3f6200-57db-11e9-9cab-c0d38ced3664.PNG)


![DRP5](https://user-images.githubusercontent.com/45924101/55646295-15ff0680-57db-11e9-8515-45d224ac2f0a.PNG)

