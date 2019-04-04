# Disaster Response Pipeline Project
## Table of Contents
1. [Installation](https://github.com/A-Nuru/Image-Classifier#Installation)
2. [Project Motivation](https://github.com/A-Nuru/Image-Classifier#Project-Motivation)
3. [File Descriptions](https://github.com/A-Nuru/Image-Classifier#File-Descriptions)
4. [Instructions](https://github.com/A-Nuru/Image-Classifier#Instructions)
5. [Results](https://github.com/A-Nuru/Image-Classifier#Results)
6. [Licensing](https://github.com/A-Nuru/Image-Classifier#Licensing)

## Installation
This project requires **Python 3.x** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org)
- [matplotlib](http://matplotlib.org/)
- [scikit-learn](http://scikit-learn.org/stable/)

It is recommended to install [Anaconda](https://www.continuum.io/downloads), a pre-packaged Python distribution that contains all of the necessary libraries and software for this project. 
You might also need to have software installed to run and execute an [iPython Notebook](http://ipython.org/notebook.htm

## Project Motivation
In this project, we analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages.The data set contains real messages that were sent during disaster events. We will build and train a machine learning pipeline to categorize these events so that the message can be sent to an appropriate disaster relief agency for prompt action.

The project includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data.


## File Descriptions
The files include:
 
- README.md

## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
